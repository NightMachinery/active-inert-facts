from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from linroute import admission
from linroute.cache import read_cache, sha256_file, slug
from linroute.probes import (
    basis_driver_snapshot,
    basis_drivers_since,
    basis_fallback_used,
    Head,
    ProbeConfig,
    Targets,
    _class_macro_mean,
    _f64,
    _i64,
    class_macro_accuracy,
    lin_cls_nested_batch,
    lin_probe_nested_batch,
    lu_probe,
    permutations,
)
from linroute.train import EXPOSURE_POLICY, LOSS_POLICY, validate_train_cache

__all__ = [
    "PRIMARY_Q",
    "TransferError",
    "RunInputs",
    "load_run_inputs",
    "manifest_fold_plan",
    "measure_transfer",
    "flat_row",
    "corruption_of",
    "corrupted_subjects",
    "rho_of",
    "rho_tag",
    "run_id",
]

PRIMARY_Q = 16

CANONICAL_TEMPLATE = 0

NAN = float("nan")

class TransferError(RuntimeError):
    pass

def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _finite(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None

def _sha(path: Path | str) -> str:
    return sha256_file(Path(path))

def _digest_folds(outer: Sequence[Sequence[int]],
                  inner: Sequence[Sequence[Sequence[int]]]) -> str:
    import hashlib

    payload = json.dumps(
        {
            "outer": [[int(i) for i in f] for f in outer],
            "inner": [[[int(i) for i in f] for f in group] for group in inner],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def read_training_cache(path: Path | str) -> dict:
    path = Path(path)
    from linroute.cache import verify_sha256

    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        verify_sha256(path)
    data = read_cache(path, validate=False)
    validate_train_cache(data)
    return data

def read_wild_cache(path: Path | str) -> dict:
    return read_cache(Path(path))

def template_split(cache: dict) -> tuple[list[int], int]:
    split = list(cache["rel_template_split"])
    train = [i for i, s in enumerate(split) if s == "train"]
    gate = [i for i, s in enumerate(split) if s == "gate"]
    if len(train) != 4 or len(gate) != 1:
        raise TransferError(
            f"rel_template_split is {split!r}: a NEW-REL arm has exactly four "
            f"'train' paraphrases and one 'gate' paraphrase"
        )
    return train, gate[0]

def _mean_over(block: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    return block[:, list(idx)].mean(1)

PRODUCTION_EPOCHS = 20

MODEL_IDS = {
    "OLMo-3-1025-7B": "allenai/Olmo-3-1025-7B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3.8-27B": "Qwen/Qwen3.8-27B",
    "Qwen3.5-9B-Base": "Qwen/Qwen3.5-9B-Base",
    "gemma-4-E2B": "google/gemma-4-E2B",
    "SmolLM3-3B-Base": "HuggingFaceTB/SmolLM3-3B-Base",
    "MiniCPM5-2B-Base": "openbmb/MiniCPM5-2B-Base",
    "G9v3-3B": "ai9stars/G9v3-3B",
    "granite-4.2-3b": "ibm-granite/granite-4.2-3b",
    "Nanbeige4-3B-Base": "Nanbeige/Nanbeige4-3B-Base",
    "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
    "LFM2.5-2.6B-Base": "LiquidAI/LFM2.5-2.6B-Base",
}

ELIGIBILITY_FIELDS = (
    "production_horizon",
    "planned_epochs",
    "epochs_done",
    "final_epoch",
    "limit_train",
    "n_train_records",
    "n_exposed_per_epoch",

    "holdout_rule",
)

REQUIRED_HOLDOUT_RULE = admission.HOLDOUT_RULE

def _exposure_per_epoch(order: dict | None) -> tuple[dict[str, int] | None, str | None]:
    if not isinstance(order, dict):
        return None, None
    epochs = order.get("epochs")
    if not isinstance(epochs, dict) or not epochs:
        return None, None
    bounds = order.get("step_bounds")
    counts: dict[str, int] = {}
    for key, perm in epochs.items():
        if not isinstance(perm, (list, tuple)):
            return None, None
        if isinstance(bounds, (list, tuple)) and bounds:
            seen: set = set()
            for pair in bounds:
                start, end = int(pair[0]), int(pair[1])
                seen.update(perm[start:end])
            counts[str(key)] = len(seen)
        else:
            counts[str(key)] = len(set(perm))
    return counts, "order.json"

def _arm_train_records(arm_dir: Path, arm: dict) -> tuple[int | None, str | None]:
    path = Path(arm_dir) / "train.jsonl"
    on_disk = None
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            on_disk = sum(1 for line in handle if line.strip())
    declared = arm.get("n_train")
    declared = int(declared) if isinstance(declared, (int, float)) else None
    if on_disk is not None and declared is not None and on_disk != declared:
        raise TransferError(
            f"{path} holds {on_disk} training records but arm.json declares "
            f"n_train {declared}: the arm directory and its own manifest of "
            f"itself disagree about what this run was trained on"
        )
    if on_disk is not None:
        return on_disk, "train.jsonl"
    if declared is not None:
        return declared, "arm.json:n_train"
    return None, None

def production_eligibility(
    *,
    planned_epochs: int | None,
    epochs_done: int | None,
    final_epoch: int | None,
    limit_train: int | None,
    n_train_records: int | None,
    exposure_counts: dict[str, int] | None,
    exposure_source: str | None,
    holdout_rule: str | None = None,
    horizon: int = PRODUCTION_EPOCHS,
    required_holdout_rule: str = REQUIRED_HOLDOUT_RULE,
) -> dict:
    reasons: list[str] = []

    def as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    planned = as_int(planned_epochs)
    done = as_int(epochs_done)
    measured = as_int(final_epoch)
    if planned is None:
        reasons.append(
            "history.config.epochs is missing: the horizon this run was "
            "LAUNCHED with cannot be established"
        )
    elif planned != horizon:
        reasons.append(
            f"planned horizon {planned} != production {horizon}: this run was "
            f"launched with a shorter recipe"
        )
    if done is None:
        reasons.append("epochs_done is missing: completion cannot be established")
    elif done != horizon:
        reasons.append(f"epochs_done {done} != production horizon {horizon}")
    if measured is None:
        reasons.append("final_epoch is missing")
    elif measured != horizon:
        reasons.append(
            f"measured at epoch {measured}, not the production horizon {horizon}"
        )

    limit = as_int(limit_train) or 0
    records = as_int(n_train_records)
    exposed_min = exposed_max = None
    if exposure_counts:
        values = [int(v) for v in exposure_counts.values()]
        exposed_min, exposed_max = min(values), max(values)
    if limit:
        reasons.append(
            f"limit_train {limit} truncates "
            + (f"{records} training records" if records is not None
               else "the arm's training records")
        )
    if records is None:
        reasons.append(
            "the arm's training-record count is unknown: exposure cannot be "
            "verified"
        )
    if exposed_min is None:
        reasons.append(
            "no per-epoch exposure is recorded (order.json absent or without "
            "step boundaries): exposure cannot be verified"
        )
    else:
        if exposed_min != exposed_max:
            reasons.append(
                f"exposure varies across epochs ({exposed_min}..{exposed_max} "
                f"distinct training records): this is not one record per epoch"
            )
        if records is not None and exposed_min != records:
            reasons.append(
                f"{exposed_min} distinct training records exposed per epoch != "
                f"the arm's {records} training records"
            )

    if holdout_rule is None:
        reasons.append(
            "no split manifest, so the held-out protocol is unknown: a run "
            "whose protocol cannot be established is not a production row"
        )
    elif holdout_rule != required_holdout_rule:
        reasons.append(
            f"protocol {admission.protocol_of(holdout_rule)}: this run's "
            f"manifest holds out by {holdout_rule!r}, not "
            f"{required_holdout_rule!r}, so its T is a different fine-tuning "
            f"set and its rows are never pooled with {required_holdout_rule}"
        )
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "production_horizon": int(horizon),
        "planned_epochs": planned,
        "epochs_done": done,
        "final_epoch": measured,
        "limit_train": limit,
        "n_train_records": records,

        "n_exposed_per_epoch": exposed_min,
        "n_exposed_per_epoch_max": exposed_max,
        "exposure_source": exposure_source,
        "exposure_per_epoch": exposure_counts,
        "holdout_rule": holdout_rule,
        "required_holdout_rule": required_holdout_rule,
        "protocol": admission.protocol_of(holdout_rule),
    }

@dataclass(frozen=True)
class RunInputs:
    run_dir: Path
    arm_dir: Path
    arm: dict
    history: dict
    artifacts: dict | None
    final_epoch: int
    planned_epochs: int | None
    epochs_done: int
    intermediate: bool
    cache0: dict
    cacheE: dict
    cache0_path: Path
    cacheE_path: Path
    preds0: list[dict]
    predsE: list[dict]
    manifest: dict | None
    manifest_path: Path | None
    wild: dict | None
    wild_path: Path | None
    files: dict

    limit_train: int | None = None
    n_train_records: int | None = None
    n_train_records_source: str | None = None
    exposure_counts: dict | None = None
    exposure_source: str | None = None
    eligibility: dict | None = None

def load_run_inputs(
    run_dir: Path | str,
    arm_dir: Path | str,
    *,
    wild_cache: Path | str | None = None,
    manifest: Path | str | None = None,
    final_epoch: int = PRODUCTION_EPOCHS,
    allow_intermediate: bool = False,
) -> RunInputs:
    run_dir = Path(run_dir)
    arm_dir = Path(arm_dir)
    history = _json(run_dir / "history.json")
    arm = _json(arm_dir / "arm.json")
    artifacts = (
        _json(run_dir / "artifacts.json")
        if (run_dir / "artifacts.json").exists() else None
    )

    epochs = sorted(int(r["epoch"]) for r in history.get("history", []))
    if not epochs:
        raise TransferError(f"{run_dir / 'history.json'} records no epoch")
    epochs_done = len([e for e in epochs if e > 0])
    planned = (history.get("config") or {}).get("epochs")
    planned = int(planned) if planned is not None else None

    final_epoch = int(final_epoch)
    missing = sorted(set(range(final_epoch + 1)) - set(epochs))
    if missing:
        raise TransferError(
            f"the run did not complete the prespecified horizon of epoch "
            f"{final_epoch}: history.json records epochs {epochs[0]}..{epochs[-1]} "
            f"and is missing {missing[:5]}{'...' if len(missing) > 5 else ''}.  "
            f"An interrupted run is a failed measurement, not a shorter one."
        )

    config = history.get("config") or {}
    limit_train = config.get("limit_train")
    order_path = run_dir / "order.json"
    order = _json(order_path) if order_path.exists() else None
    exposure_counts, exposure_source = _exposure_per_epoch(order)
    n_train_records, n_train_records_source = _arm_train_records(arm_dir, arm)

    intermediate = planned is not None and final_epoch != planned
    if intermediate and not allow_intermediate:
        raise TransferError(
            f"--final-epoch {final_epoch} is not this run's planned horizon "
            f"({planned}).  Measuring an intermediate epoch is allowed, but it "
            f"must be asked for explicitly and its row is never 'complete'."
        )

    states = run_dir / "states"
    name = slug(history.get("arm") or arm.get("arm") or "")
    path0 = states / f"{name}_ep000.pt"
    pathE = states / f"{name}_ep{final_epoch:03d}.pt"
    for path in (path0, pathE):
        if not path.exists():
            raise TransferError(f"missing state cache {path}")

    cache0 = read_training_cache(path0)
    cacheE = read_training_cache(pathE)
    preds0 = _jsonl(run_dir / "preds_ep000.jsonl")
    predsE = _jsonl(run_dir / f"preds_ep{final_epoch:03d}.jsonl")

    files = {
        "history": _sha(run_dir / "history.json"),
        "arm": _sha(arm_dir / "arm.json"),
        "cache_ep000": _sha(path0),
        f"cache_ep{final_epoch:03d}": _sha(pathE),
        "preds_ep000": _sha(run_dir / "preds_ep000.jsonl"),
        f"preds_ep{final_epoch:03d}": _sha(run_dir / f"preds_ep{final_epoch:03d}.jsonl"),
    }
    if order is not None:
        files["order"] = _sha(order_path)

    man = None
    if manifest is not None:
        man = _json(Path(manifest))
        files["manifest"] = _sha(manifest)
    wild = None
    if wild_cache is not None:
        wild = read_wild_cache(wild_cache)
        files["wild_cache"] = _sha(wild_cache)

    return RunInputs(
        run_dir=run_dir, arm_dir=arm_dir, arm=arm, history=history,
        artifacts=artifacts, planned_epochs=planned, epochs_done=epochs_done,
        intermediate=intermediate,
        limit_train=limit_train,
        n_train_records=n_train_records,
        n_train_records_source=n_train_records_source,
        exposure_counts=exposure_counts, exposure_source=exposure_source,
        eligibility=production_eligibility(
            planned_epochs=planned, epochs_done=epochs_done,
            final_epoch=final_epoch, limit_train=limit_train,
            n_train_records=n_train_records, exposure_counts=exposure_counts,
            exposure_source=exposure_source,
            holdout_rule=admission.holdout_rule_of(man),
        ),
        final_epoch=int(final_epoch), cache0=cache0, cacheE=cacheE,
        cache0_path=path0, cacheE_path=pathE, preds0=preds0, predsE=predsE,
        manifest=man, manifest_path=Path(manifest) if manifest else None,
        wild=wild, wild_path=Path(wild_cache) if wild_cache else None,
        files=files,
    )

def verify_provenance(
    inputs: RunInputs,
    *,
    model_name: str,
    relation: str,
    repeat: int,
    core: bool,
) -> dict:
    arm, history = inputs.arm, inputs.history
    config = history.get("config") or {}
    meta = inputs.cacheE["meta"]
    mismatches: list[str] = []
    MISSING = object()

    def dig(root: dict, *path: str):
        node = root
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return MISSING
            node = node[key]
        return node

    def require(label: str, value, *, allow_falsy: bool = False):
        if value is MISSING or value is None or (not allow_falsy and value == ""):
            mismatches.append(f"{label}: missing from the run's artifacts")
            return MISSING
        return value

    def want(label: str, got, expected) -> None:
        if got is MISSING or expected is MISSING:
            return
        if got is not None and expected is not None and got != expected:
            mismatches.append(f"{label}: run says {got!r}, caller says {expected!r}")

    seeds = arm.get("seeds") or {}

    slug_of_arm = arm.get("relation_slug")

    expected_id = MODEL_IDS.get(model_name)
    run_model = require("history.config.model", dig(config, "model"))
    if run_model is MISSING:
        run_model = meta.get("model")
    if expected_id and run_model not in (MISSING, None) and run_model != expected_id:
        mismatches.append(
            f"model: the run trained {run_model!r}, --model-name says "
            f"{model_name!r} ({expected_id!r})"
        )
    want("cache model", meta.get("model"), config.get("model"))
    require("cache meta.model_revision", meta.get("model_revision"))

    for label, policy in (("cache meta", meta.get("exposure_policy")),
                          ("history", history.get("exposure_policy"))):
        got = require(f"{label}.exposure_policy", policy)
        if got is not MISSING and got != EXPOSURE_POLICY:
            mismatches.append(
                f"{label}.exposure_policy is {got!r}, not {EXPOSURE_POLICY!r}: "
                f"this run did not see every training record once per epoch"
            )

    for label, policy in (("cache meta", meta.get("loss_policy")),
                          ("history", history.get("loss_policy"))):
        if policy is not None and policy != LOSS_POLICY:
            mismatches.append(
                f"{label}.loss_policy is {policy!r}, not {LOSS_POLICY!r}: "
                f"this run's loss was taken over a different set of positions"
            )

    stalled = [
        int(r["epoch"]) for r in history.get("history", [])
        if isinstance(r.get("weight_change"), dict)
        and int(r["epoch"]) > 0
        and (r["weight_change"].get("steps") or 0) > 0
        and (r["weight_change"].get("changed_params") or 0) == 0
    ]
    if stalled:
        mismatches.append(
            f"epochs {stalled[:6]} took optimizer steps and changed no "
            f"parameter: their cached states are not what they claim"
        )

    want("arm.repeat", require("arm.repeat", arm.get("repeat"),
                               allow_falsy=True), repeat)
    want("arm.seeds.repeat",
         require("arm.seeds.repeat", dig(seeds, "repeat"), allow_falsy=True), repeat)
    want("arm.seeds.optimizer_seed",
         require("arm.seeds.optimizer_seed", dig(seeds, "optimizer_seed"),
                 allow_falsy=True), repeat)
    bijection_key = require("arm.seeds.bijection_key", dig(seeds, "bijection_key"))
    if bijection_key is not MISSING and slug_of_arm:
        want("arm.seeds.bijection_key", bijection_key,
             f"bijection|{slug_of_arm}|{repeat}")
    want("history.config.seed",
         require("history.config.seed", dig(config, "seed"), allow_falsy=True), repeat)

    require("arm.arm", arm.get("arm"))
    if history.get("arm") and arm.get("arm") and history["arm"] != arm["arm"]:
        mismatches.append(
            f"arm: the run trained {history['arm']!r}, --arm-dir holds "
            f"{arm['arm']!r}"
        )
    arm_meta = history.get("arm_meta")
    if isinstance(arm_meta, dict) and arm_meta and arm_meta != arm:
        differing = sorted(
            k for k in set(arm_meta) | set(arm) if arm_meta.get(k) != arm.get(k)
        )
        mismatches.append(
            f"arm identity: history.json's arm_meta differs from "
            f"{inputs.arm_dir / 'arm.json'} in {differing[:6]}"
        )
    if inputs.artifacts is not None:
        a_config = inputs.artifacts.get("config") or {}
        for key in ("model", "seed", "epochs", "lr"):
            want(f"artifacts.json {key}", a_config.get(key), config.get(key))

    declared = require("arm.animals", arm.get("animals"))
    class_animal = require("arm.class_animal", arm.get("class_animal"))
    if declared is not MISSING and class_animal is not MISSING:
        if sorted(class_animal.values()) != sorted(declared):
            mismatches.append(
                f"arm.class_animal binds {sorted(set(class_animal.values()))[:4]}..., "
                f"which is not the declared animal subset"
            )
        if len(set(class_animal.values())) != len(class_animal):
            mismatches.append("arm.class_animal is not a bijection: an animal repeats")
        cache_candidates = list(inputs.cacheE["candidates"])
        if list(declared) != cache_candidates:
            mismatches.append(
                f"arm.animals is not the cache's candidate axis: "
                f"{list(declared)[:3]} vs {cache_candidates[:3]}"
            )

    assertion = require("arm.animal_assertion", arm.get("animal_assertion"))
    if assertion is not MISSING and assertion.get("passed") is not True:
        mismatches.append(
            f"arm.animal_assertion.passed is {assertion.get('passed')!r}: the "
            f"single-token answer property was not verified for this arm"
        )

    want("arm.relation_slug", require("arm.relation_slug", slug_of_arm), relation)
    arm_display = arm.get("relation")
    if inputs.manifest is not None:
        manifest_display = require("manifest.relation",
                                   inputs.manifest.get("relation"))
        if manifest_display is not MISSING:
            derived = slug(str(manifest_display))
            if derived.casefold() != relation.casefold():
                mismatches.append(
                    f"manifest.relation {manifest_display!r} slugifies to "
                    f"{derived!r}, not the requested relation {relation!r}"
                )
            if arm_display and str(arm_display).casefold() != str(manifest_display).casefold():
                mismatches.append(
                    f"relation name: the arm was built for {arm_display!r}, the "
                    f"manifest is {manifest_display!r}"
                )
        declared_sha = require("arm.manifest.sha256", dig(arm, "manifest", "sha256"))
        if declared_sha is not MISSING and inputs.files.get("manifest") != declared_sha:
            mismatches.append(
                f"manifest: the arm was built from sha256 {declared_sha}, "
                f"--manifest is {inputs.files.get('manifest')}"
            )
        checks_passed = require("arm.manifest.checks_passed",
                                dig(arm, "manifest", "checks_passed"))
        if checks_passed is not MISSING and checks_passed is not True:
            mismatches.append(
                f"arm.manifest.checks_passed is {checks_passed!r}: the arm was "
                f"built from a manifest that failed its own audit"
            )

    verdict = {
        "core": bool(core and not mismatches),
        "verified": not mismatches,
        "mismatches": mismatches,
        "run_model": None if run_model is MISSING else run_model,
        "model_revision": meta.get("model_revision"),
        "optimizer_seed": seeds.get("optimizer_seed", config.get("seed")),
        "bijection_key": seeds.get("bijection_key"),
        "animal_subset_key": seeds.get("animal_subset_key"),
        "planned_epochs": inputs.planned_epochs,
        "epochs_done": inputs.epochs_done,
        "arm": arm.get("arm"),
        "arm_repeat": arm.get("repeat"),

        "relation_slug": slug_of_arm,
        "relation_name": arm_display,
        "manifest_relation": (inputs.manifest or {}).get("relation"),
        "arm_manifest_sha256": dig(arm, "manifest", "sha256")
        if dig(arm, "manifest", "sha256") is not MISSING else None,
        "animal_assertion_passed": (arm.get("animal_assertion") or {}).get("passed"),
        "exposure_policy": meta.get("exposure_policy"),
        "history_exposure_policy": history.get("exposure_policy"),
        "weight_change_per_epoch": {
            str(r["epoch"]): r["weight_change"].get("changed_params")
            for r in history.get("history", [])
            if isinstance(r.get("weight_change"), dict)
        } or None,
        "epochs_with_no_weight_change": stalled,
    }
    if mismatches and core:
        raise TransferError(
            "the run's own artifacts contradict, or do not carry, the identity "
            "this row would claim: " + "; ".join(mismatches)
        )
    return verdict

def manifest_fold_plan(
    manifest: dict, subjects: Sequence[str]
) -> tuple[list[np.ndarray], list[list[np.ndarray]]]:
    if manifest.get("schema") != admission.SCHEMA:
        raise TransferError(
            f"manifest schema is {manifest.get('schema')!r}, not {admission.SCHEMA!r}"
        )
    if not manifest.get("checks", {}).get("passed"):
        raise TransferError(
            "the split manifest did not pass its own admission/fold audit "
            f"({manifest.get('checks', {}).get('failures')}); it is not usable "
            f"as a fold plan"
        )

    by_id = {s["subject_id"]: s["subject"] for s in manifest["train"]["subjects"]}
    row = {name: i for i, name in enumerate(subjects)}
    if len(row) != len(subjects):
        raise TransferError("the state matrix has a repeated subject string")

    def positions(ids: Sequence[str]) -> np.ndarray:
        out = []
        for sid in ids:
            if sid not in by_id:
                raise TransferError(f"manifest fold names subject id {sid!r}, not in T")
            name = by_id[sid]
            if name not in row:
                raise TransferError(
                    f"manifest subject {name!r} has no row in the state cache"
                )
            out.append(row[name])
        return np.sort(np.asarray(out, dtype=np.int64))

    outer: list[np.ndarray] = []
    inner: list[list[np.ndarray]] = []
    for fold in manifest["folds"]["outer"]:
        outer.append(positions(fold["validation"]))
        inner.append([positions(sub["validation"]) for sub in fold["inner"]])
    return outer, inner

def _class_map(arm: dict) -> dict[str, str]:
    table = arm.get("class_animal") or {}
    if not table:
        return {}
    out: dict[str, str] = {}
    for key, animal in table.items():
        if animal in out:
            raise TransferError(
                f"class_animal maps two classes onto animal {animal!r}: not a bijection"
            )
        out[animal] = str(key)
    return out

def _classes_of(rows: Sequence[dict], cmap: dict[str, str]) -> np.ndarray:
    return np.asarray([cmap.get(r["target"], r["target"]) for r in rows], dtype=object)

def _macro(correct: Sequence[float], classes: np.ndarray) -> float:
    correct = np.asarray(correct, dtype=np.float64)
    if not len(correct):
        return NAN
    return float(np.mean([correct[classes == c].mean() for c in np.unique(classes)]))

def _select(rows: Sequence[dict], group: str, templates: set[str]) -> list[dict]:
    return [r for r in rows if r["group"] == group and r.get("template") in templates]

def behavioural_outcomes(
    preds: Sequence[dict],
    preds0: Sequence[dict],
    arm: dict,
    *,
    n_classes: int,
    expected_heldout: dict[str, str] | None = None,
    holdout_per_class: dict | None = None,
    holdout_rule: str | None = None,
) -> dict:
    cmap = _class_map(arm)
    templates = arm.get("templates") or {}
    train_names = set(templates.get("train") or [])
    gate_names = set(templates.get("gate") or [])
    if len(train_names) != 4 or len(gate_names) != 1:
        raise TransferError(
            f"arm.json declares {len(train_names)} train and {len(gate_names)} "
            f"gate templates; a NEW-REL arm has four and one"
        )

    held = _select(preds, "heldout", gate_names)
    held_classes = _classes_of(held, cmap)
    n_held_classes = len(np.unique(held_classes))
    counts = {}
    for c in held_classes:
        counts[str(c)] = counts.get(str(c), 0) + 1
    if holdout_per_class is None:
        if n_held_classes != n_classes:
            raise TransferError(
                f"H under the evaluation template covers {n_held_classes} of "
                f"{n_classes} effective classes; every class must be present"
            )
    else:
        want = {str(c): int(n) for c, n in holdout_per_class.items()}
        if counts != want:
            wrong = sorted(
                set(want) | set(counts),
                key=lambda c: (counts.get(c, 0) - want.get(c, 0), c),
            )
            detail = [(c, counts.get(c, 0), want.get(c, 0)) for c in wrong
                      if counts.get(c, 0) != want.get(c, 0)]
            raise TransferError(
                f"H under the evaluation template has {len(held)} rows over "
                f"{n_held_classes} classes, but the manifest's split holds out "
                f"{sum(want.values())} over {len(want)}: (class, got, wanted) "
                f"{detail[:4]}.  The per-class counts ARE the measurement: a "
                f"run that lost one subject of a large class and gained one of "
                f"a small class passes every total and reports a different "
                f"class-macro number under the same name."
            )
    if expected_heldout is not None:
        got = {r["town"] for r in held}
        if got != set(expected_heldout):
            raise TransferError(
                f"the evaluation-template rows cover {sorted(got)[:4]}..., not "
                f"the H subjects the caches and the manifest agree on "
                f"({sorted(expected_heldout)[:4]}...)"
            )
        wrong = [
            (r["town"], r["target"], expected_heldout[r["town"]])
            for r in held if r["target"] != expected_heldout[r["town"]]
        ]
        if wrong:
            raise TransferError(
                f"{len(wrong)} held-out rows carry a gold animal that is not "
                f"the one this subject's frozen class is bound to, e.g. {wrong[0]}"
            )

    seen = _select(preds, "gate", gate_names)
    recall = _select(preds, "train", train_names)
    ep0_held = _select(preds0, "heldout", gate_names)

    by_subject: dict[str, list[float]] = {}
    subject_class: dict[str, str] = {}
    for r in recall:
        by_subject.setdefault(r["town"], []).append(float(r["correct"]))
        subject_class[r["town"]] = cmap.get(r["target"], r["target"])
    recall_subjects = sorted(by_subject)
    recall_values = [float(np.mean(by_subject[s])) for s in recall_subjects]
    recall_classes = np.asarray([subject_class[s] for s in recall_subjects], dtype=object)

    held_correct = [float(r["correct"]) for r in held]
    seen_correct = [float(r["correct"]) for r in seen]
    return {
        "heldout_cand_acc": _macro(held_correct, held_classes),
        "heldout_cand_acc_micro": float(np.mean(held_correct)),
        "seen_cand_acc": _macro(seen_correct, _classes_of(seen, cmap)),
        "seen_cand_acc_micro": float(np.mean(seen_correct)) if seen_correct else NAN,
        "train_recall_acc": _macro(recall_values, recall_classes),
        "train_recall_acc_micro": (
            float(np.mean([float(r["correct"]) for r in recall])) if recall else NAN
        ),
        "newrel_gen_ep0_acc": (
            _macro([float(r["correct"]) for r in ep0_held], _classes_of(ep0_held, cmap))
            if ep0_held else NAN
        ),
        "chance": 1.0 / float(n_classes),
        "n_heldout_rows": len(held),
        "n_seen_rows": len(seen),
        "n_train_rows": len(recall),
        "n_train_subjects": len(recall_subjects),
        "holdout_rule": holdout_rule,
        "protocol": admission.protocol_of(holdout_rule),
        "holdout_per_class": (dict(counts) if holdout_per_class is None
                              else {str(c): int(n)
                                    for c, n in holdout_per_class.items()}),
        "heldout_acc_is_class_macro": True,
        "evaluation_template": sorted(gate_names)[0],
    }

def _probe_block(
    S: np.ndarray,
    targets: Targets,
    head: Head,
    cfg: ProbeConfig,
    *,
    q: int,
    folds=None,
    inner_folds=None,
    n_perm: int = 0,
) -> dict:
    batch = [targets]
    if n_perm:
        perms = permutations(len(targets.gold_token), n_perm, cfg.perm_seed)
        batch += [targets.permuted(p) for p in perms]
    fits_before = basis_driver_snapshot()
    results = lin_probe_nested_batch(
        S, batch, head, cfg, (q,), folds=folds, inner_folds=inner_folds
    )
    real = results[0][q]
    out = {
        "available": bool(real["available"]),

        "basis_drivers": basis_drivers_since(fits_before),
        "acc": real["acc"],
        "base_acc": real["base_acc"],
        "gain": real["delta"],
        "majority_acc": real["majority_acc"],
        "freq_acc": real["freq_acc"],
        "micro_acc": real["micro_acc"],
        "micro_base_acc": real["micro_base_acc"],
        "micro_gain": real["micro_delta"],
        "layers": list(real.get("layers") or []),
        "modal_layer": real.get("modal_layer"),
        "n": real["n"],
        "q": int(q),
    }
    if not real["available"]:
        out["reason"] = real.get("reason")
    if n_perm:
        shuffled = [r[q]["delta"] for r in results[1:]]
        out["shuffle_gain_mean"] = float(np.mean(shuffled))
        out["shuffle_gain_std"] = float(np.std(shuffled))
        out["shuffle_gain_values"] = [float(v) for v in shuffled]
    return out

def _cls_block(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "pred"}

def _secondary_block(name: str, compute, failures: list[str]) -> dict:
    try:
        return compute()
    except (np.linalg.LinAlgError, ArithmeticError) as exc:
        failures.append(name)
        err = f"{type(exc).__name__}: {exc}"
        return {"available": False, "failed": True, "error": err,
                "reason": f"{name} failed: {err}"}

def measure_transfer(
    inputs: RunInputs,
    *,
    model_name: str,
    relation: str,
    repeat: int,
    cfg: ProbeConfig | None = None,
    q: int = PRIMARY_Q,
    n_perm: int = 5,
    rank_check: str = "outer",
) -> dict:
    cfg = cfg or ProbeConfig(q_primary=q, n_perm=n_perm)

    backend = cfg.make_backend()
    c0, cE = inputs.cache0, inputs.cacheE
    checks: dict = {"files_sha256": dict(inputs.files)}

    eligibility = inputs.eligibility
    if eligibility is None:
        eligibility = production_eligibility(
            planned_epochs=inputs.planned_epochs, epochs_done=inputs.epochs_done,
            final_epoch=inputs.final_epoch, limit_train=inputs.limit_train,
            n_train_records=inputs.n_train_records,
            exposure_counts=inputs.exposure_counts,
            exposure_source=inputs.exposure_source,
        )
    checks["eligibility"] = {
        **eligibility,
        "n_train_records_source": inputs.n_train_records_source,
    }

    core_requested = inputs.manifest is not None and inputs.wild is not None
    core_requested = core_requested and not inputs.intermediate
    core_requested = core_requested and bool(eligibility["eligible"])
    checks["provenance"] = verify_provenance(
        inputs, model_name=model_name, relation=relation, repeat=repeat,
        core=core_requested,
    )
    core = bool(checks["provenance"]["core"]) and bool(eligibility["eligible"])
    checks["core_requirements"] = {
        "manifest": inputs.manifest is not None,
        "wild_cache": inputs.wild is not None,
        "horizon_completed": not inputs.intermediate,

        "production_horizon": all(
            eligibility.get(k) == eligibility["production_horizon"]
            for k in ("planned_epochs", "epochs_done", "final_epoch")
        ),
        "training_exposure_complete": (
            not eligibility["limit_train"]
            and eligibility["n_train_records"] is not None
            and eligibility["n_exposed_per_epoch"] is not None
            and eligibility["n_exposed_per_epoch"]
            == eligibility["n_exposed_per_epoch_max"]
            == eligibility["n_train_records"]
        ),
        "provenance_verified": checks["provenance"]["verified"],
    }

    if list(c0["subjects"]) != list(cE["subjects"]):
        raise TransferError(
            "the epoch-0 and epoch-E caches do not carry the same subjects in "
            "the same order; s_0 and t_e would be paired across different rows"
        )
    for key in ("subject_split", "subject_region", "candidates",
                "rel_templates", "rel_template_names", "rel_template_split"):
        if list(c0[key]) != list(cE[key]):
            raise TransferError(
                f"{key} differs between the epoch-0 and epoch-E caches: the "
                f"two files do not describe the same experiment"
            )
    if not np.array_equal(_i64(c0["cand_index"]), _i64(cE["cand_index"])):
        raise TransferError("cand_index differs between the two epochs' caches")
    if not np.array_equal(_i64(c0["cand_first_token"]), _i64(cE["cand_first_token"])):
        raise TransferError("cand_first_token differs between the two epochs' caches")
    checks["subject_order_identical"] = True
    checks["caches_agree"] = True
    checks["schema_versions"] = {
        "cache_ep000": c0["meta"]["schema_version"],
        "cache_epE": cE["meta"]["schema_version"],
    }

    checks["fidelity_passed"] = {
        name: _fidelity_verdict(cache["meta"]["fidelity"])
        for name, cache in (("cache_ep000", c0), ("cache_epE", cE))
    }
    if not all(v["passed"] for v in checks["fidelity_passed"].values()):
        raise TransferError(
            "a training cache's pre-norm fidelity check did not pass: its "
            "answer states do not decode to what the model itself chose"
        )

    split = list(c0["subject_split"])
    subjects = list(c0["subjects"])
    t_rows = [i for i, s in enumerate(split) if s == "train"]
    h_rows = [i for i, s in enumerate(split) if s == "heldout"]
    if not t_rows:
        raise TransferError("no subject has subject_split == 'train'")

    manifest = inputs.manifest
    if manifest is not None:
        order = [s["subject"] for s in manifest["train"]["subjects"]]
        pos = {subjects[i]: i for i in t_rows}
        missing = [name for name in order if name not in pos]
        if missing:
            raise TransferError(
                f"{len(missing)} manifest T subjects are absent from the "
                f"training cache's train split, e.g. {missing[:3]}"
            )
        extra = sorted(set(pos) - set(order))
        if extra:
            raise TransferError(
                f"{len(extra)} cache train subjects are not in the manifest's "
                f"T, e.g. {extra[:3]}"
            )
        t_rows = [pos[name] for name in order]
    t_names = [subjects[i] for i in t_rows]
    h_names = [subjects[i] for i in h_rows]
    checks["heldout_excluded_from_fits"] = not (set(t_names) & set(h_names))
    if not checks["heldout_excluded_from_fits"]:
        raise TransferError("a subject is in both T and H")

    labels_all = _i64(cE["cand_index"])
    animals = list(cE["candidates"])
    class_of_subject = _manifest_classes(manifest)
    gold_animal = {
        subjects[i]: animals[int(labels_all[i])] for i in range(len(subjects))
    }
    if manifest is not None:
        want_h = {s["subject"] for s in manifest["holdout"]["subjects"]}
        if set(h_names) != want_h:
            raise TransferError(
                f"H in the caches is not the manifest's H: "
                f"{sorted(set(h_names) ^ want_h)[:4]} differ"
            )
        bound = _class_to_animal(inputs.arm, animals)
        if bound is not None:
            corrupt = corruption_of(inputs.arm)
            skipped = corrupted_subjects(inputs.arm)
            wrong = [
                (name, gold_animal[name], bound[class_of_subject[name]])
                for name in t_names + h_names
                if name in class_of_subject
                and name not in skipped
                and gold_animal[name] != bound[class_of_subject[name]]
            ]
            if wrong:
                raise TransferError(
                    f"{len(wrong)} subjects carry a gold animal that is not the "
                    f"one their frozen effective class is bound to by the arm's "
                    f"bijection, e.g. {wrong[0]}"
                )
            checks["gold_classes_match_manifest"] = True
            if corrupt is not None:
                checks["gold_classes_skipped_corrupt"] = len(skipped)
                realised = corrupt.get("realised_changed_fraction")
                checks["corrupt_realised_changed_fraction"] = (
                    None if realised is None else float(realised)
                )

        else:
            checks["gold_classes_match_manifest"] = None
        checks["heldout_identity_matches_manifest"] = True

    train_tpl, gate_tpl = template_split(cE)
    if template_split(c0) != (train_tpl, gate_tpl):
        raise TransferError("the two epochs' caches disagree about which template is the gate")
    names = list(cE["rel_template_names"])
    checks["templates"] = {
        "fitting": [names[i] for i in train_tpl],
        "gate_excluded": names[gate_tpl],
        "fitting_indices": list(train_tpl),
        "gate_index": int(gate_tpl),
    }

    rows = np.asarray(t_rows, dtype=np.int64)
    S0 = _mean_over(_f64(c0["rel_subject_last"]), train_tpl)[rows]
    SE = _mean_over(_f64(cE["rel_subject_last"]), train_tpl)[rows]

    S0_first = (_mean_over(_f64(c0["rel_subject_first"]), train_tpl)[rows]
                if "rel_subject_first" in c0 else None)
    t0 = _mean_over(_f64(c0["rel_answer_final"]), train_tpl)[rows]
    tE = _mean_over(_f64(cE["rel_answer_final"]), train_tpl)[rows]

    head0, headE = Head.from_cache(c0), Head.from_cache(cE)
    labels = _i64(cE["cand_index"])[rows]
    gold = headE.cand_token[labels]
    targets_e = Targets(T=tE, gold_token=gold, label=labels)
    targets_0 = Targets(T=t0, gold_token=head0.cand_token[labels], label=labels)

    folds = inner = None
    if manifest is not None:
        folds, inner = manifest_fold_plan(manifest, t_names)
        checks["folds"] = {
            "source": "split_manifest",
            "schema": manifest["schema"],
            "n_outer": len(folds),
            "n_inner": [len(g) for g in inner],
            "sha256": _digest_folds(folds, inner),
            "manifest_checks_passed": True,
        }
        checks["centered_rank_scope"] = rank_check
        checks["centered_rank"] = _rank_report(S0, manifest, q=q, scope=rank_check)
        checks["centered_rank_all_layers_feasible"] = (
            all(v["passed"] for v in checks["centered_rank"].values())
            if checks["centered_rank"] else None
        )
    else:
        checks["folds"] = {
            "source": "internal_seeded",
            "n_outer": cfg.outer_folds,
            "n_inner": cfg.inner_folds,
            "folds_seed": cfg.folds_seed,
            "sha256": None,
        }

    kw = dict(folds=folds, inner_folds=inner, q=q)

    fits_at_start = basis_driver_snapshot()
    secondary_failures: list[str] = []
    primary = _probe_block(S0, targets_e, headE, cfg, n_perm=n_perm, **kw)
    se_te = _secondary_block(
        "head_se_te", lambda: _probe_block(SE, targets_e, headE, cfg, **kw),
        secondary_failures)
    ctrl_newrel = _secondary_block(
        "head_control_newrel_s0_t0",
        lambda: _probe_block(S0, targets_0, head0, cfg, **kw),
        secondary_failures)

    if S0_first is None:
        primary_first = {
            "available": False,
            "reason": "the epoch-0 state cache carries no rel_subject_first "
                      "(written before the first-token states were cached)",
        }
    else:
        primary_first = _secondary_block(
            "primary_probe_first",
            lambda: _probe_block(S0_first, targets_e, headE, cfg, **kw),
            secondary_failures)
        primary_first["subject_position"] = "first"

    n_classes = len(cE["candidates"])

    holdout_rule = admission.holdout_rule_of(manifest)
    want_per_class = None
    if manifest is not None:
        by_class = admission.per_class_holdout(manifest)
        bound = _class_to_animal(inputs.arm, animals)
        cmap = _class_map(inputs.arm)
        if bound is not None:
            want_per_class = {
                cmap.get(bound[c], str(c)): n for c, n in by_class.items()
                if c in bound
            }
        else:
            want_per_class = {str(c): n for c, n in by_class.items()}
    behaviour = behavioural_outcomes(
        inputs.predsE, inputs.preds0, inputs.arm, n_classes=n_classes,
        expected_heldout={name: gold_animal[name] for name in h_names},
        holdout_per_class=want_per_class, holdout_rule=holdout_rule,
    )

    if want_per_class is not None:
        cache_counts: dict[int, int] = {}
        for i in h_rows:
            c = class_of_subject.get(subjects[i])
            if c is not None:
                cache_counts[c] = cache_counts.get(c, 0) + 1
        want_ids = {int(c): int(n)
                    for c, n in admission.per_class_holdout(manifest).items()}
        if cache_counts != want_ids:
            raise TransferError(
                f"the caches hold out {sum(cache_counts.values())} subjects "
                f"over {len(cache_counts)} classes, but the manifest's split "
                f"holds out {sum(want_ids.values())} over {len(want_ids)}"
            )
    checks["holdout_rule"] = holdout_rule
    checks["protocol"] = admission.protocol_of(holdout_rule)
    checks["holdout_per_class_matches_manifest"] = want_per_class is not None

    status = "complete" if core else "partial"
    reason = None
    if not core:
        unmet = [k for k, v in checks["core_requirements"].items() if not v]
        reason = "not a core row: " + ", ".join(unmet)

        if eligibility["reasons"]:
            reason += "; " + "; ".join(eligibility["reasons"])

    rho = rho_of(inputs.arm)
    protocol = eligibility["protocol"]
    if rho is not None and protocol is not None:
        protocol = f"{protocol}-corrupt"
    metrics: dict = {
        "schema": "linroute/transfer-metrics/2",
        "status": status,
        "core": core,
        "status_reason": reason,
        "model": model_name,
        "relation": relation,
        "repeat": int(repeat),
        "final_epoch": inputs.final_epoch,
        "planned_epochs": inputs.planned_epochs,

        "epochs_done": inputs.epochs_done,

        "production_horizon": eligibility["production_horizon"],
        "limit_train": eligibility["limit_train"],
        "n_train_records": eligibility["n_train_records"],
        "n_exposed_per_epoch": eligibility["n_exposed_per_epoch"],

        "holdout_rule": eligibility["holdout_rule"],
        "protocol": protocol,
        "holdout_per_class": want_per_class,
        "production_eligible": bool(eligibility["eligible"]),
        "arm": inputs.arm.get("arm"),
        "n_train": len(t_rows),
        "n_heldout": len(h_rows),
        "n_classes": n_classes,
        "q": int(q),
        "primary_probe": primary,

        "primary_probe_first": primary_first,
        "head_se_te": se_te,
        "head_control_newrel_s0_t0": ctrl_newrel,
        "behaviour": behaviour,
        "checks": checks,
    }
    if rho is not None:
        metrics["rho"] = rho
        metrics["rho_pct"] = int(round(rho * 100))
    metrics["support_profile"] = _support_profile(
        labels_all, n_classes, manifest=manifest
    )
    metrics["support_profile_train"] = _support_profile(labels, n_classes)
    metrics.update(_provenance(inputs, repeat, manifest))

    if inputs.wild is not None:
        metrics["pre_rel"] = _secondary_block(
            "pre_rel",
            lambda: _pre_rel_block(
                inputs.wild, t_names, cfg, q=q, folds=folds, inner_folds=inner,
                checks=checks, heldout_names=h_names,
            ),
            secondary_failures)
    else:
        metrics["pre_rel"] = {"available": False,
                              "reason": "no wild cache supplied"}

    metrics["secondary_failures"] = list(secondary_failures)
    metrics["basis_drivers"] = basis_drivers_since(fits_at_start)
    metrics["basis_fallback_used"] = basis_fallback_used(metrics["basis_drivers"])

    metrics.update(backend.describe())
    return metrics

_SERIALIZED_COUNTS = (
    "recon_cand_match", "recon_cand_mismatches", "recon_tie_mismatches",
    "recon_nontie_mismatches", "recon_max_chosen_regret",
    "recon_regret_mismatches", "serial_cand_match", "serial_cand_mismatches",
    "serial_tie_mismatches", "serial_nontie_mismatches",
    "serial_max_chosen_regret", "serial_regret_mismatches",
)

def _fidelity_verdict(fidelity: dict) -> dict:
    out = {"passed": bool(fidelity.get("passed"))}
    for block in ("live", "serialized"):
        node = fidelity.get(block)
        if not isinstance(node, dict):
            continue
        out[block] = {
            "passed": bool(node.get("passed")),
            "rel_diff": node.get("rel_diff"),
            **({"argmax_match": node["argmax_match"]}
               if "argmax_match" in node else {}),
            **{k: node[k] for k in _SERIALIZED_COUNTS if k in node},
        }
    return out

def _manifest_classes(manifest: dict | None) -> dict[str, int]:
    if manifest is None:
        return {}
    return {
        s["subject"]: int(s["effective_class"])
        for key in ("train", "holdout")
        for s in manifest[key]["subjects"]
    }

def corruption_of(arm: dict) -> dict | None:
    block = arm.get("corruption")
    return block if isinstance(block, dict) and block else None

def corrupted_subjects(arm: dict) -> frozenset[str]:
    block = corruption_of(arm)
    if block is None:
        return frozenset()
    return frozenset(str(name) for name in (block.get("selected") or ()))

def rho_of(arm: dict) -> float | None:
    block = corruption_of(arm)
    if block is None:
        return None
    rho = block.get("rho")
    return None if rho is None else float(rho)

def rho_tag(rho: float) -> str:
    return f"rho{int(round(float(rho) * 100)):03d}"

def _class_to_animal(arm: dict, animals: Sequence[str]) -> dict[int, str] | None:
    table = arm.get("class_animal") or {}
    if not table:
        return None
    out: dict[int, str] = {}
    for key, animal in table.items():
        if animal not in animals:
            raise TransferError(
                f"class_animal binds class {key} to {animal!r}, which is not a "
                f"candidate of the cache"
            )
        out[int(key)] = animal
    return out

def _provenance(inputs: RunInputs, repeat: int, manifest: dict | None) -> dict:
    arm, meta = inputs.arm, inputs.cacheE["meta"]
    params = (manifest or {}).get("parameters", {})
    wild_meta = (inputs.wild or {}).get("meta", {})
    seeds = arm.get("seeds") or {}

    per_tokenizer = (arm.get("animal_assertion") or {}).get("tokenizers") or []
    revisions = {
        str(t.get("tokenizer")): t.get("revision")
        for t in per_tokenizer if isinstance(t, dict)
    }
    return {
        "arm_sha256": inputs.files.get("arm"),
        "manifest_sha256": inputs.files.get("manifest"),
        "arm_manifest_sha256": (arm.get("manifest") or {}).get("sha256")
        if isinstance(arm.get("manifest"), dict) else None,
        "relation_slug": arm.get("relation_slug"),
        "relation_name": arm.get("relation"),
        "relation_type": arm.get("relation_type") or meta.get("relation_type"),
        "relation_source": arm.get("relation_source"),
        "model_revision": meta.get("model_revision"),
        "tokenizer_revision": revisions.get(str(meta.get("tokenizer")))
        or (next(iter(revisions.values())) if len(revisions) == 1 else None),
        "tokenizer_revisions": revisions or None,
        "n_merged_classes": wild_meta.get("n_merged_classes"),

        "seeds": {
            "role": seeds.get("role_seed", params.get("role_seed")),
            "fold": seeds.get("manifest_fold_seed", params.get("fold_seed")),
            "bijection": seeds.get("repeat", int(repeat)),
            "bijection_key": seeds.get("bijection_key"),
            "animal_subset_key": seeds.get("animal_subset_key"),
            "optimizer": seeds.get("optimizer_seed", int(repeat)),
        },
    }

def _rank_report(
    S0: np.ndarray, manifest: dict, *, q: int, scope: str
) -> dict[str, dict]:
    if scope == "none":
        return {}
    if scope not in ("outer", "all"):
        raise ValueError(f"rank_check must be none, outer or all; got {scope!r}")
    view = manifest
    if scope == "outer":
        view = {
            **manifest,
            "folds": {
                **manifest["folds"],
                "outer": [{**f, "inner": []} for f in manifest["folds"]["outer"]],
            },
        }
    return {
        f"layer_{layer}": admission.check_centered_rank(S0[:, layer, :], view, q=q)
        for layer in range(S0.shape[1])
    }

def _support_profile(
    labels: np.ndarray, n_classes: int, *, manifest: dict | None = None
) -> list[int]:
    if manifest is not None:
        retained = [c for c in manifest["classes"] if c.get("retained")]
        if retained:
            return [int(c["support"]) for c in retained]
    counts = np.bincount(labels, minlength=n_classes)
    return [int(c) for c in counts if c > 0]

def _pre_rel_block(
    wild: dict,
    t_names: Sequence[str],
    cfg: ProbeConfig,
    *,
    q: int,
    folds,
    inner_folds,
    checks: dict,
    heldout_names: Sequence[str] = (),
) -> dict:
    subjects = list(wild["subjects"])
    pos = {name: i for i, name in enumerate(subjects)}
    missing = [name for name in t_names if name not in pos]
    if missing:
        raise TransferError(
            f"{len(missing)} of {len(t_names)} T subjects are absent from the "
            f"PRE-REL wild cache, e.g. {missing[:3]}.  The PRE-REL measures "
            f"would then describe a different subject set from the probe's."
        )
    leaked = sorted(set(t_names) & set(heldout_names))
    if leaked:
        raise TransferError(
            f"{len(leaked)} held-out subjects are among the T rows the PRE-REL "
            f"block would fit, e.g. {leaked[:3]}.  H never enters a fit."
        )
    rows = np.asarray([pos[name] for name in t_names], dtype=np.int64)

    head = Head.from_cache(wild)
    raw_labels = _i64(wild["cand_index"])[rows]

    present, labels = np.unique(raw_labels, return_inverse=True)
    n_classes = int(len(present))
    gold = head.cand_token[raw_labels]

    t = CANONICAL_TEMPLATE
    n_templates = len(wild["rel_templates"])
    if t >= n_templates:
        raise TransferError("the wild cache has no canonical template")
    S = _f64(wild["rel_subject_last"])[rows][:, t]
    T_ans = _f64(wild["rel_answer_final"])[rows][:, t]
    layer_axis = S.shape[1]
    L = layer_axis - 1

    checks["pre_rel"] = {
        "schema_version": wild["meta"]["schema_version"],
        "token_rule": wild["meta"].get("token_rule"),
        "decode_fidelity_passed": bool(wild["meta"]["decode_fidelity"]["passed"]),
        "canonical_template_index": t,
        "canonical_template": wild["rel_templates"][t],
        "n_matched": int(len(rows)),
        "n_wild_subjects": len(subjects),
        "n_classes_in_T": n_classes,
    }

    cls = lin_cls_nested_batch(
        S, [labels], n_classes, cfg, (q,), folds=folds, inner_folds=inner_folds
    )[0][q]
    control = _probe_block(
        S, Targets(T=T_ans, gold_token=gold, label=labels), head, cfg,
        q=q, folds=folds, inner_folds=inner_folds,
    )

    out: dict = {
        "available": True,
        "n_matched": int(len(rows)),
        "n_classes": n_classes,
        "cls": _cls_block(cls),
        "head_control_prerel_s0_t0": control,
    }

    if "rel_gold_object_mean" in wild:
        lu_subject_layer = L // 2 + 1
        lu_object_layer = max(0, L - 1)
        S_lu = _f64(wild["rel_subject_mean"])[rows][:, t, lu_subject_layer, :]
        O_lu = _f64(wild["rel_gold_object_mean"])[rows][:, t, lu_object_layer, :]
        lu = lu_probe(
            S_lu, O_lu, folds=cfg.lu_folds, folds_seed=cfg.folds_seed,
            lam=cfg.lu_lam, train_frac=cfg.lu_train_frac,
            backend=cfg.make_backend(),
        )
        out["lu"] = {
            **lu,
            "subject_layer": int(lu_subject_layer),
            "object_layer": int(lu_object_layer),

            "affine_minus_const": lu["delta_cos_affine_cv"] - lu["delta_cos_const_cv"],
        }
    else:
        out["lu"] = {"available": False,
                     "reason": "the wild cache has no teacher-forced object states"}

    zs = _i64(wild["zs_cand"])[rows][:, t]
    hit = head.tokens(zs) == gold
    out["cand_acc"] = _class_macro_mean(hit, gold)
    out["cand_acc_micro"] = float(hit.mean())
    head_pred = head.decode(T_ans)
    out["head_acc"] = class_macro_accuracy(head.tokens(head_pred), gold)
    return out

def run_id(model_name: str, relation: str, repeat: int,
           rho: float | None = None) -> str:
    rid = f"{slug(model_name)}/{slug(relation)}/r{int(repeat)}"
    return rid if rho is None else f"{rid}/{rho_tag(rho)}"

def flat_row(metrics: dict, *, git_commit: str, status: str | None = None) -> dict:
    status = status or metrics.get("status") or "complete"
    primary = metrics["primary_probe"]
    se_te = metrics["head_se_te"]
    ctrl = metrics["head_control_newrel_s0_t0"]
    beh = metrics["behaviour"]
    pre = metrics.get("pre_rel") or {}
    pre_ok = bool(pre.get("available"))
    cls = pre.get("cls") or {}
    lu = pre.get("lu") or {}
    prerel_ctrl = pre.get("head_control_prerel_s0_t0") or {}

    first_probe = metrics.get("primary_probe_first") or {}
    first_available = bool(first_probe.get("available"))

    rho = metrics.get("rho")
    rid = run_id(metrics["model"], metrics["relation"], metrics["repeat"], rho)
    row = {
        "status": status,
        "model": metrics["model"],
        "relation": metrics["relation"],
        "repeat": int(metrics["repeat"]),

        "rho": rho,
        "primary_probe_gain": _finite(primary.get("gain")),
        "heldout_cand_acc": _finite(beh.get("heldout_cand_acc")),
        "seen_cand_acc": _finite(beh.get("seen_cand_acc")),
        "n_train": metrics["n_train"],

        "n_heldout": metrics["n_heldout"],
        "n_classes": metrics["n_classes"],
        "holdout_rule": metrics.get("holdout_rule"),
        "protocol": metrics.get("protocol"),
        "holdout_per_class": metrics.get("holdout_per_class"),
        "provenance": f"{rid}@{git_commit}",
        "run_id": rid,
        "primary_probe_acc": _finite(primary.get("acc")),
        "primary_probe_base_acc": _finite(primary.get("base_acc")),
        "primary_probe_layers": list(primary.get("layers") or []),

        "primary_probe_first_gain": (
            _finite(first_probe.get("gain")) if first_available else None),
        "primary_probe_first_acc": (
            _finite(first_probe.get("acc")) if first_available else None),
        "primary_probe_first_base_acc": (
            _finite(first_probe.get("base_acc")) if first_available else None),
        "primary_probe_first_layers": (
            list(first_probe.get("layers") or []) if first_available else None),
        "primary_probe_first_available": first_available,
        "head_se_te_gain": _finite(se_te.get("gain")),
        "head_se_te_acc": _finite(se_te.get("acc")),
        "head_se_te_base_acc": _finite(se_te.get("base_acc")),
        "head_control_newrel_s0_t0_gain": _finite(ctrl.get("gain")),
        "head_control_prerel_s0_t0_gain": (
            _finite(prerel_ctrl.get("gain")) if pre_ok else None
        ),
        "pre_rel_cls_acc": _finite(cls.get("acc")) if pre_ok else None,
        "pre_rel_cls_majority_acc": _finite(cls.get("majority_acc")) if pre_ok else None,
        "pre_rel_cls_gain": _finite(cls.get("gain")) if pre_ok else None,
        "pre_rel_lu_affine_minus_const": (
            _finite(lu.get("affine_minus_const")) if pre_ok else None
        ),
        "pre_rel_cand_acc": _finite(pre.get("cand_acc")) if pre_ok else None,
        "pre_rel_cand_acc_micro": _finite(pre.get("cand_acc_micro")) if pre_ok else None,
        "train_recall_acc": _finite(beh.get("train_recall_acc")),
        "newrel_gen_ep0_acc": _finite(beh.get("newrel_gen_ep0_acc")),
        "chance": _finite(beh.get("chance")),
        "heldout_cand_acc_micro": _finite(beh.get("heldout_cand_acc_micro")),
        "seen_cand_acc_micro": _finite(beh.get("seen_cand_acc_micro")),
        "shuffle_gain_mean": _finite(primary.get("shuffle_gain_mean")),
        "shuffle_gain_std": _finite(primary.get("shuffle_gain_std")),
        "support_profile": metrics.get("support_profile"),
        "final_epoch": metrics.get("final_epoch"),
        "epochs_done": metrics.get("epochs_done"),
        "config_hash": metrics["checks"]["folds"].get("sha256"),

        "basis_drivers": metrics.get("basis_drivers"),
        "basis_fallback_used": metrics.get("basis_fallback_used"),

        "compute_backend": metrics.get("compute_backend"),
        "compute_device": metrics.get("compute_device"),
        "compute_dtype": metrics.get("compute_dtype"),
        "secondary_failures": list(metrics.get("secondary_failures") or []),
        "error": None,
    }
    for key in ("seeds", "manifest_sha256", "arm_sha256", "model_revision",
                "tokenizer_revision", "n_merged_classes", "relation_type",
                "relation_source", "core", "status_reason", "planned_epochs",
                "support_profile_train",

                "production_horizon", "limit_train", "n_train_records",
                "n_exposed_per_epoch", "production_eligible"):
        row[key] = metrics.get(key)
    row["error"] = metrics.get("status_reason") if status != "complete" else None
    return row

def failed_row(
    *,
    model_name: str,
    relation: str,
    repeat: int,
    git_commit: str,
    error: str,
    status: str = "failed",
    holdout_rule: str | None = None,
    rho: float | None = None,
) -> dict:
    rid = run_id(model_name, relation, repeat, rho)
    return {
        "status": status,
        "model": model_name,
        "relation": relation,
        "repeat": int(repeat),
        "rho": rho,
        "primary_probe_gain": None,
        "heldout_cand_acc": None,
        "seen_cand_acc": None,
        "n_train": None,
        "n_heldout": None,
        "n_classes": None,

        "holdout_rule": holdout_rule,
        "protocol": admission.protocol_of(holdout_rule),
        "holdout_per_class": None,
        "provenance": f"{rid}@{git_commit}",
        "run_id": rid,
        "error": error,
    }
