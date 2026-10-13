from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .probes import (
    Head,
    ProbeConfig,
    Targets,
    _f64,
    _i64,
    class_macro_accuracy,
    lin_probe_nested_batch,
)
from .transfer import (
    RunInputs,
    _digest_folds,
    _mean_over,
    manifest_fold_plan,
    template_split,
)

class CorruptionAnalysisError(ValueError):
    pass

@dataclass(frozen=True)
class E3Counts:
    assigned: int
    original: int
    other: int

    @property
    def total(self) -> int:
        return self.assigned + self.original + self.other

def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def _unique_names(values: Sequence[object], *, field: str) -> list[str]:
    names = [str(value) for value in values]
    if len(names) != len(set(names)):
        raise CorruptionAnalysisError(f"{field} contains duplicate subjects")
    return names

def e3_cell(arm: dict, predictions: Iterable[dict]) -> dict:
    corruption = arm.get("corruption")
    if not isinstance(corruption, dict):
        raise CorruptionAnalysisError("arm has no corruption block")

    selected = _unique_names(corruption.get("selected") or [], field="selected")
    changed = _unique_names(corruption.get("changed") or [], field="changed")
    if not changed:
        raise CorruptionAnalysisError("changed-subject denominator is empty")
    selected_set = set(selected)
    changed_set = set(changed)
    if not changed_set <= selected_set:
        raise CorruptionAnalysisError("changed subjects are not a subset of selected")
    if int(corruption.get("n_selected", -1)) != len(selected):
        raise CorruptionAnalysisError("n_selected disagrees with selected")
    if int(corruption.get("n_changed", -1)) != len(changed):
        raise CorruptionAnalysisError("n_changed disagrees with changed")

    clean = corruption.get("clean_label")
    assigned = corruption.get("corrupt_label")
    if not isinstance(clean, dict) or not isinstance(assigned, dict):
        raise CorruptionAnalysisError("clean_label/corrupt_label maps are missing")
    if set(clean) != set(assigned):
        raise CorruptionAnalysisError("clean and corrupt label maps have different subjects")
    n_train = int(corruption.get("n_train_subjects", -1))
    if len(clean) != n_train:
        raise CorruptionAnalysisError(
            f"label maps contain {len(clean)} subjects, expected n_train_subjects={n_train}"
        )

    derived_changed = {subject for subject in clean if clean[subject] != assigned[subject]}
    if changed_set != derived_changed:
        omitted = sorted(derived_changed - changed_set)
        spurious = sorted(changed_set - derived_changed)
        raise CorruptionAnalysisError(
            "stored changed set disagrees with labels: "
            f"omitted={omitted!r}, spurious={spurious!r}"
        )

    candidates = arm.get("animals")
    if not isinstance(candidates, list) or not candidates:
        raise CorruptionAnalysisError("arm has no candidate animal list")
    candidate_set = {str(candidate) for candidate in candidates}
    if len(candidate_set) != len(candidates):
        raise CorruptionAnalysisError("candidate animal list contains duplicates")
    for subject in clean:
        for kind, label in (("clean", clean[subject]), ("assigned", assigned[subject])):
            if label not in candidate_set:
                raise CorruptionAnalysisError(
                    f"{subject!r} {kind} label {label!r} is outside the candidate set"
                )
    for subject in changed:
        if clean[subject] == assigned[subject]:
            raise CorruptionAnalysisError(
                f"changed subject {subject!r} has identical clean and assigned labels"
            )

    templates = arm.get("templates") or {}
    train_templates = list((templates.get("train") or {}).keys())
    if len(train_templates) != 4 or len(set(train_templates)) != 4:
        raise CorruptionAnalysisError(
            f"arm declares {len(train_templates)} distinct training templates, expected 4"
        )
    wanted_templates = set(train_templates)

    rows_by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        subject = str(row.get("town"))
        if row.get("group") == "train" and subject in changed_set:
            rows_by_subject[subject].append(row)

    category = Counter()
    per_subject = Counter()
    for subject in changed:
        rows = rows_by_subject.get(subject, [])
        names = [str(row.get("template")) for row in rows]
        if len(rows) != 4 or set(names) != wanted_templates or len(names) != len(set(names)):
            raise CorruptionAnalysisError(
                f"{subject!r} has templates {sorted(names)!r}, expected "
                f"{sorted(wanted_templates)!r} exactly once"
            )
        local = Counter()
        for row in rows:
            if row.get("target") != assigned[subject]:
                raise CorruptionAnalysisError(
                    f"{subject!r}/{row.get('template')} target {row.get('target')!r} "
                    f"is not assigned label {assigned[subject]!r}"
                )
            pred = row.get("pred")
            if pred not in candidate_set:
                raise CorruptionAnalysisError(
                    f"{subject!r}/{row.get('template')} prediction {pred!r} "
                    "is outside the candidate set"
                )
            if pred == assigned[subject]:
                label = "assigned"
            elif pred == clean[subject]:
                label = "original"
            else:
                label = "other"
            category[label] += 1
            local[label] += 1
        if sum(local.values()) != 4:
            raise AssertionError("the four-template trichotomy is not exhaustive")
        per_subject.update({f"n_{name}_{count}": 1 for name, count in local.items()})

    counts = E3Counts(
        assigned=category["assigned"],
        original=category["original"],
        other=category["other"],
    )
    expected = 4 * len(changed)
    if counts.total != expected:
        raise AssertionError(f"trichotomy has {counts.total} trials, expected {expected}")

    return {
        "rho": float(corruption["rho"]),
        "rho_pct": int(corruption["rho_pct"]),
        "n_train_subjects": int(corruption["n_train_subjects"]),
        "n_selected_subjects": len(selected),
        "n_changed_subjects": len(changed),
        "n_templates": 4,
        "n_trials": counts.total,
        "assigned_count": counts.assigned,
        "original_count": counts.original,
        "other_count": counts.other,
        "assigned_rate": counts.assigned / counts.total,
        "original_rate": counts.original / counts.total,
        "other_rate": counts.other / counts.total,
        "selected_fraction": float(corruption["selected_fraction"]),
        "realised_changed_fraction": float(corruption["realised_changed_fraction"]),
        "category_contract": "changed-subject x four-training-template",
    }

def conditional_class_label_alignment(
    arm: dict,
    predictions: Iterable[dict],
    *,
    seed: int,
    monte_carlo_draws: int = 100_000,
    exact_max_permutations: int = 40_320,
) -> dict:
    if monte_carlo_draws <= 0:
        raise CorruptionAnalysisError("monte_carlo_draws must be positive")

    candidates = arm.get("animals")
    class_animal = arm.get("class_animal")
    subjects = arm.get("subjects")
    if not isinstance(candidates, list) or not candidates:
        raise CorruptionAnalysisError("arm has no candidate animal list")
    if not isinstance(class_animal, dict) or not class_animal:
        raise CorruptionAnalysisError("arm has no class_animal bijection")
    if not isinstance(subjects, list) or not subjects:
        raise CorruptionAnalysisError("arm has no subject records")

    candidate_set = {str(candidate) for candidate in candidates}
    if len(candidate_set) != len(candidates):
        raise CorruptionAnalysisError("candidate animal list contains duplicates")
    clean_map = {str(key): str(value) for key, value in class_animal.items()}
    if len(set(clean_map.values())) != len(clean_map):
        raise CorruptionAnalysisError("class_animal is not one-to-one")
    if set(clean_map.values()) != candidate_set:
        raise CorruptionAnalysisError(
            "class_animal values do not equal the candidate animal set"
        )
    if int(arm.get("n_classes", -1)) != len(clean_map):
        raise CorruptionAnalysisError(
            "arm n_classes disagrees with class_animal/candidate count"
        )

    heldout_class: dict[str, str] = {}
    for record in subjects:
        if record.get("split") != "heldout":
            continue
        subject = str(record.get("town"))
        class_key = str(record.get("effective_class"))
        if subject in heldout_class:
            raise CorruptionAnalysisError(f"duplicate held-out subject {subject!r}")
        if class_key not in clean_map:
            raise CorruptionAnalysisError(
                f"held-out subject {subject!r} has unknown class {class_key!r}"
            )
        heldout_class[subject] = class_key
    if len(heldout_class) != int(arm.get("n_heldout_subjects", -1)):
        raise CorruptionAnalysisError(
            "held-out subject records disagree with n_heldout_subjects"
        )

    gate = list(((arm.get("templates") or {}).get("gate") or {}).keys())
    if len(gate) != 1:
        raise CorruptionAnalysisError(
            f"arm declares {len(gate)} gate templates, expected one"
        )
    gate_name = gate[0]
    rows = [
        row for row in predictions
        if row.get("group") == "heldout" and row.get("template") == gate_name
    ]
    row_subjects = [str(row.get("town")) for row in rows]
    if len(row_subjects) != len(set(row_subjects)):
        raise CorruptionAnalysisError("held-out gate predictions repeat a subject")
    if set(row_subjects) != set(heldout_class):
        omitted = sorted(set(heldout_class) - set(row_subjects))
        extra = sorted(set(row_subjects) - set(heldout_class))
        raise CorruptionAnalysisError(
            f"held-out gate prediction identities disagree: omitted={omitted!r}, "
            f"extra={extra!r}"
        )

    class_keys = sorted(clean_map)
    class_position = {key: index for index, key in enumerate(class_keys)}
    clean_animals = [clean_map[key] for key in class_keys]
    animal_position = {animal: index for index, animal in enumerate(clean_animals)}
    counts = np.zeros((len(class_keys), len(class_keys)), dtype=np.int64)
    n_by_class = np.zeros(len(class_keys), dtype=np.int64)
    for row in rows:
        subject = str(row["town"])
        class_key = heldout_class[subject]
        expected_target = clean_map[class_key]
        if row.get("target") != expected_target:
            raise CorruptionAnalysisError(
                f"{subject!r} target {row.get('target')!r} is not clean label "
                f"{expected_target!r}"
            )
        pred = str(row.get("pred"))
        if pred not in animal_position:
            raise CorruptionAnalysisError(
                f"{subject!r} prediction {pred!r} is outside the candidate set"
            )
        stored_correct = row.get("correct")
        expected_correct = int(pred == expected_target)
        if isinstance(stored_correct, bool):
            stored_correct = int(stored_correct)
        if stored_correct not in (0, 1) or int(stored_correct) != expected_correct:
            raise CorruptionAnalysisError(
                f"{subject!r} stored correct={row.get('correct')!r} disagrees "
                f"with pred==target ({expected_correct})"
            )
        i = class_position[class_key]
        counts[i, animal_position[pred]] += 1
        n_by_class[i] += 1
    if np.any(n_by_class == 0):
        absent = [class_keys[i] for i in np.flatnonzero(n_by_class == 0)]
        raise CorruptionAnalysisError(f"held-out predictions omit classes {absent!r}")

    proportions = counts / n_by_class[:, None]
    identity = np.arange(len(class_keys), dtype=np.int64)

    def statistic(assignment: np.ndarray) -> float:
        return float(np.mean(proportions[identity, assignment]))

    observed = statistic(identity)
    total_permutations = math.factorial(len(class_keys))
    tolerance = 1e-15
    if total_permutations <= exact_max_permutations:
        null = np.asarray(
            [statistic(np.asarray(perm, dtype=np.int64))
             for perm in permutations(range(len(class_keys)))],
            dtype=np.float64,
        )
        exceed = int(np.count_nonzero(null >= observed - tolerance))
        p_value = exceed / len(null)
        method = "exact-enumeration"
        correction = "none-exact"
        draws = int(len(null))
    else:
        rng = np.random.default_rng(seed)
        null = np.empty(monte_carlo_draws, dtype=np.float64)
        for draw in range(monte_carlo_draws):
            null[draw] = statistic(rng.permutation(len(class_keys)))
        exceed = int(np.count_nonzero(null >= observed - tolerance))
        p_value = (exceed + 1) / (monte_carlo_draws + 1)
        method = "seeded-monte-carlo"
        correction = "plus-one"
        draws = monte_carlo_draws

    chance = 1.0 / len(class_keys)
    return {
        "observed_class_macro": observed,
        "chance": chance,
        "observed_minus_chance": observed - chance,
        "p_one_sided_above_chance": float(p_value),
        "null_draws_at_least_observed": exceed,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "null_min": float(np.min(null)),
        "null_max": float(np.max(null)),
        "n_effective_classes": len(class_keys),
        "n_heldout_subjects": len(rows),
        "heldout_per_class": {
            key: int(n_by_class[class_position[key]]) for key in class_keys
        },
        "method": method,
        "draws": draws,
        "total_possible_permutations": total_permutations,
        "p_value_correction": correction,
        "seed": int(seed),
        "unit": "effective-class",
        "conditioning": (
            "observed per-class heldout prediction vectors and candidate set"
        ),
        "null": "clean animal labels exchangeable across effective classes",
        "interpretation": (
            "exploratory residual alignment with the clean mapping; not causal rule reuse"
        ),
        "exact_design_based_randomization_test": False,
    }

def validate_e1_endpoint_identity(
    arm: dict,
    metrics: dict,
    *,
    model_name: str,
    relation: str,
) -> None:
    corruption = arm.get("corruption")
    if not isinstance(corruption, dict):
        raise CorruptionAnalysisError("E1 endpoint arm has no corruption block")
    if not np.isclose(float(corruption.get("rho", -1)), 1.0, atol=0, rtol=0):
        raise CorruptionAnalysisError("E1 endpoint arm corruption rho is not 1")
    if int(corruption.get("rho_pct", -1)) != 100:
        raise CorruptionAnalysisError("E1 endpoint arm rho_pct is not 100")
    if int(arm.get("repeat", -1)) != 0:
        raise CorruptionAnalysisError("E1 endpoint arm repeat is not 0")
    if arm.get("relation_slug") != relation:
        raise CorruptionAnalysisError("E1 endpoint arm relation does not match key")

    required = {
        "status": "complete",
        "core": True,
        "production_eligible": True,
        "final_epoch": 20,
        "repeat": 0,
        "rho_pct": 100,
        "model": model_name,
        "relation": relation,
    }
    for key, expected in required.items():
        if metrics.get(key) != expected:
            raise CorruptionAnalysisError(
                f"E1 endpoint metrics {key}={metrics.get(key)!r}, "
                f"expected {expected!r}"
            )
    if not np.isclose(float(metrics.get("rho", -1)), 1.0, atol=0, rtol=0):
        raise CorruptionAnalysisError("E1 endpoint metrics rho is not 1")

def rule_rescore(
    pred_token: Sequence[int],
    base_token: Sequence[int],
    clean_gold_token: Sequence[int],
) -> dict:
    pred = np.asarray(pred_token, dtype=np.int64)
    base = np.asarray(base_token, dtype=np.int64)
    gold = np.asarray(clean_gold_token, dtype=np.int64)
    if pred.ndim != 1 or base.shape != pred.shape or gold.shape != pred.shape:
        raise CorruptionAnalysisError(
            "prediction, baseline, and gold must be equal 1D vectors"
        )
    if not len(gold):
        raise CorruptionAnalysisError("rule rescore has no subjects")
    acc = class_macro_accuracy(pred, gold)
    base_acc = class_macro_accuracy(base, gold)
    return {
        "acc": acc,
        "base_acc": base_acc,
        "gain": acc - base_acc,
        "micro_acc": float(np.mean(pred == gold)),
        "micro_base_acc": float(np.mean(base == gold)),
        "micro_gain": float(np.mean(pred == gold) - np.mean(base == gold)),
        "n": int(len(gold)),
        "scoring": "same-as-run-predictions-original-rule-gold",
    }

def _parity_float(name: str, recovered: float, archived: float) -> None:
    if not np.isclose(float(recovered), float(archived), atol=1e-12, rtol=0):
        raise CorruptionAnalysisError(
            f"primary parity failed for {name}: recovered={recovered!r}, "
            f"archived={archived!r}"
        )

def primary_probe_parity(recovered: dict, archived: dict) -> dict:
    if not recovered.get("available") or not archived.get("available"):
        raise CorruptionAnalysisError("primary probe is unavailable")
    for key in ("acc", "base_acc", "micro_acc", "micro_base_acc"):
        _parity_float(key, recovered[key], archived[key])
    _parity_float("gain", recovered["delta"], archived["gain"])
    _parity_float("micro_gain", recovered["micro_delta"], archived["micro_gain"])
    for key in ("n", "q", "modal_layer"):
        if int(recovered[key]) != int(archived[key]):
            raise CorruptionAnalysisError(
                f"primary parity failed for {key}: recovered={recovered[key]!r}, "
                f"archived={archived[key]!r}"
            )
    if list(recovered.get("layers") or []) != list(archived.get("layers") or []):
        raise CorruptionAnalysisError("primary parity failed for selected layers")
    return {
        "passed": True,
        "absolute_tolerance": 1e-12,
        "fields": [
            "acc", "base_acc", "gain", "micro_acc", "micro_base_acc",
            "micro_gain", "n", "q", "modal_layer", "layers",
        ],
    }

def heuristic_gain_normalization(raw_gain: float, changed_fraction: float) -> dict:
    raw = float(raw_gain)
    changed = float(changed_fraction)
    if not 0 <= changed <= 1:
        raise CorruptionAnalysisError(f"changed fraction {changed!r} is outside [0, 1]")
    denominator = 1.0 - changed
    if denominator == 0:
        value = amplification = None
        status = "unavailable-c-equals-one"
    else:
        value = raw / denominator
        amplification = 1.0 / denominator
        status = (
            "available-near-singular" if denominator <= 0.1 else "available"
        )
    return {
        "raw_gain": raw,
        "realised_changed_fraction_c": changed,
        "denominator_one_minus_c": denominator,
        "corruption_normalized_heuristic_gain": value,
        "amplification_factor": amplification,
        "near_singular": 0 < denominator <= 0.1,
        "near_singular_threshold": 0.1,
        "status": status,
        "formula": "g_raw / (1 - c)",
        "example": ".20/(1-.50)=.40",
        "reader_label": "corruption-normalized (heuristic)",
        "historical_trace_label": "ceiling-corrected",
        "is_proven_ceiling": False,
        "is_accuracy": False,
    }

def validate_p4b_identity(
    inputs: RunInputs,
    metrics: dict,
    *,
    model_name: str,
    relation: str,
    rho_pct: int,
) -> dict:
    corruption = inputs.arm.get("corruption")
    if not isinstance(corruption, dict):
        raise CorruptionAnalysisError("P4b input arm has no corruption block")
    expected_rho = rho_pct / 100.0
    if not np.isclose(float(corruption.get("rho", -1)), expected_rho, atol=0, rtol=0):
        raise CorruptionAnalysisError("P4b arm rho does not match path key")
    if int(corruption.get("rho_pct", -1)) != rho_pct:
        raise CorruptionAnalysisError("P4b arm rho_pct does not match path key")
    if int(inputs.arm.get("repeat", -1)) != 0:
        raise CorruptionAnalysisError("P4b arm repeat is not 0")
    if inputs.arm.get("relation_slug") != relation:
        raise CorruptionAnalysisError("P4b arm relation does not match key")
    required = {
        "status": "complete",
        "core": True,
        "production_eligible": True,
        "final_epoch": 20,
        "repeat": 0,
        "rho_pct": rho_pct,
        "model": model_name,
        "relation": relation,
        "git_commit": "0861cf0cfe3cd7fbae622fbcae42abef11d7f4e8",
        "protocol": "heldout-mushroom-v2-corrupt",
        "holdout_rule": "mushroom-v2",
    }
    for key, expected in required.items():
        if metrics.get(key) != expected:
            raise CorruptionAnalysisError(
                f"P4b metrics {key}={metrics.get(key)!r}, expected {expected!r}"
            )
    if not np.isclose(float(metrics.get("rho", -1)), expected_rho, atol=0, rtol=0):
        raise CorruptionAnalysisError("P4b metrics rho does not match path key")
    if inputs.arm.get("git_commit") != "030c85b4e0c05f7bef70371153d43769e85b0b1e":
        raise CorruptionAnalysisError(
            f"P4b arm git_commit={inputs.arm.get('git_commit')!r}, expected frozen "
            "030c85b4e0c05f7bef70371153d43769e85b0b1e"
        )

    archived_files = ((metrics.get("checks") or {}).get("files_sha256") or {})
    required_files = (
        "history", "arm", "cache_ep000", "cache_ep020", "preds_ep000",
        "preds_ep020", "order", "manifest",
    )
    for key in required_files:
        recovered = inputs.files.get(key)
        archived = archived_files.get(key)
        if recovered is None or archived is None or recovered != archived:
            raise CorruptionAnalysisError(
                f"P4b source hash mismatch for {key}: "
                f"recovered={recovered!r}, archived={archived!r}"
            )
    return {
        "metrics_git_commit": metrics["git_commit"],
        "arm_git_commit": inputs.arm["git_commit"],
        "protocol": metrics["protocol"],
        "holdout_rule": metrics["holdout_rule"],
        "source_hashes_match_archived_metrics": True,
    }

def validated_corruption_labels(
    order: Sequence[str],
    manifest: dict,
    arm: dict,
    candidates: Sequence[str],
    assigned_index: Sequence[int],
) -> dict:
    order = [str(name) for name in order]
    candidate_list = [str(candidate) for candidate in candidates]
    assigned_index = np.asarray(assigned_index, dtype=np.int64)
    if len(order) != len(assigned_index):
        raise CorruptionAnalysisError("assigned index vector does not match T")
    if assigned_index.size and (
        int(assigned_index.min()) < 0 or int(assigned_index.max()) >= len(candidate_list)
    ):
        raise CorruptionAnalysisError("assigned index is outside candidate axis")

    records = manifest.get("train", {}).get("subjects") or []
    by_subject = {str(record["subject"]): record for record in records}
    if len(by_subject) != len(records) or set(by_subject) != set(order):
        raise CorruptionAnalysisError("manifest T records do not exactly match T order")
    class_animal = {
        str(key): str(value) for key, value in (arm.get("class_animal") or {}).items()
    }
    if not class_animal:
        raise CorruptionAnalysisError("arm has no clean class_animal bijection")
    derived_clean = []
    for name in order:
        class_key = str(by_subject[name]["effective_class"])
        if class_key not in class_animal:
            raise CorruptionAnalysisError(
                f"manifest class {class_key!r} has no class_animal binding"
            )
        derived_clean.append(class_animal[class_key])
    derived_assigned = [candidate_list[int(index)] for index in assigned_index]

    corruption = arm.get("corruption")
    if not isinstance(corruption, dict):
        raise CorruptionAnalysisError("arm has no corruption block")
    stored_clean = corruption.get("clean_label")
    stored_assigned = corruption.get("corrupt_label")
    if not isinstance(stored_clean, dict) or set(stored_clean) != set(order):
        raise CorruptionAnalysisError("stored clean labels do not exactly cover T")
    if not isinstance(stored_assigned, dict) or set(stored_assigned) != set(order):
        raise CorruptionAnalysisError("stored assigned labels do not exactly cover T")
    for name, clean, assigned in zip(order, derived_clean, derived_assigned):
        if stored_clean[name] != clean:
            raise CorruptionAnalysisError(
                f"clean-rule mismatch for {name!r}: manifest/bijection={clean!r}, "
                f"stored={stored_clean[name]!r}"
            )
        if stored_assigned[name] != assigned:
            raise CorruptionAnalysisError(
                f"assigned-cache mismatch for {name!r}: cache={assigned!r}, "
                f"stored={stored_assigned[name]!r}"
            )

    derived_changed = {
        name for name, clean, assigned in zip(order, derived_clean, derived_assigned)
        if clean != assigned
    }
    changed_list = list(map(str, corruption.get("changed") or []))
    selected_list = list(map(str, corruption.get("selected") or []))
    if len(changed_list) != len(set(changed_list)):
        raise CorruptionAnalysisError("stored changed list contains duplicates")
    if len(selected_list) != len(set(selected_list)):
        raise CorruptionAnalysisError("stored selected list contains duplicates")
    stored_changed = set(changed_list)
    selected = set(selected_list)
    if not selected <= set(order):
        raise CorruptionAnalysisError("selected subjects are outside T")
    if stored_changed != derived_changed:
        raise CorruptionAnalysisError("stored changed set disagrees with derived labels")
    if not derived_changed <= selected:
        raise CorruptionAnalysisError("derived changed subjects are not selected")
    if int(corruption.get("n_changed", -1)) != len(derived_changed):
        raise CorruptionAnalysisError("stored n_changed disagrees with derived labels")
    if int(corruption.get("n_selected", -1)) != len(selected):
        raise CorruptionAnalysisError("stored n_selected disagrees with selected set")
    changed_fraction = len(derived_changed) / len(order)
    if not np.isclose(
        float(corruption.get("realised_changed_fraction", -1)),
        changed_fraction,
        atol=1e-15,
        rtol=0,
    ):
        raise CorruptionAnalysisError(
            "stored realised changed fraction disagrees with derived labels"
        )
    return {
        "clean_animals": derived_clean,
        "assigned_animals": derived_assigned,
        "changed_subjects": sorted(derived_changed),
        "n_changed": len(derived_changed),
        "realised_changed_fraction": changed_fraction,
    }

def recover_primary_rule_rescore(
    inputs: RunInputs,
    archived_metrics: dict,
    *,
    q: int = 16,
    torch_device: str | None = None,
) -> dict:
    if inputs.manifest is None:
        raise CorruptionAnalysisError("P4b requires the run's frozen manifest")
    if archived_metrics.get("q") != q:
        raise CorruptionAnalysisError(
            f"archived metrics q={archived_metrics.get('q')!r}, expected {q}"
        )
    archived_backend = archived_metrics.get("compute_backend")
    archived_device = archived_metrics.get("compute_device")
    archived_dtype = archived_metrics.get("compute_dtype")
    if archived_backend not in ("numpy", "torch") or archived_dtype != "float64":
        raise CorruptionAnalysisError(
            f"unsupported archived backend tuple "
            f"{(archived_backend, archived_device, archived_dtype)!r}"
        )
    if archived_backend == "numpy":
        if archived_device != "cpu":
            raise CorruptionAnalysisError("archived numpy primary is not on CPU")
        recovery_device = "cpu"
    else:
        if not str(archived_device).startswith("cuda:"):
            raise CorruptionAnalysisError("archived torch primary is not on CUDA")
        if torch_device is None or not torch_device.startswith("cuda:"):
            raise CorruptionAnalysisError(
                "archived torch/CUDA primary requires an explicit recovery CUDA device"
            )
        recovery_device = torch_device

    c0, cE = inputs.cache0, inputs.cacheE
    if list(c0["subjects"]) != list(cE["subjects"]):
        raise CorruptionAnalysisError("epoch-0 and epoch-20 subject order differs")
    for key in (
        "subject_split", "subject_region", "candidates", "rel_templates",
        "rel_template_names", "rel_template_split",
    ):
        if list(c0[key]) != list(cE[key]):
            raise CorruptionAnalysisError(f"epoch caches disagree on {key}")
    if not np.array_equal(_i64(c0["cand_index"]), _i64(cE["cand_index"])):
        raise CorruptionAnalysisError("epoch caches disagree on cand_index")
    if not np.array_equal(
        _i64(c0["cand_first_token"]), _i64(cE["cand_first_token"])
    ):
        raise CorruptionAnalysisError("epoch caches disagree on candidate tokens")

    subjects = list(c0["subjects"])
    split = list(c0["subject_split"])
    cache_train = {subjects[i]: i for i, value in enumerate(split) if value == "train"}
    order = [record["subject"] for record in inputs.manifest["train"]["subjects"]]
    if set(cache_train) != set(order):
        raise CorruptionAnalysisError("cache and manifest training identities differ")
    rows = np.asarray([cache_train[name] for name in order], dtype=np.int64)

    train_templates, _ = template_split(cE)
    if template_split(c0) != template_split(cE):
        raise CorruptionAnalysisError("epoch caches disagree on template split")
    S0 = _mean_over(_f64(c0["rel_subject_last"]), train_templates)[rows]
    tE = _mean_over(_f64(cE["rel_answer_final"]), train_templates)[rows]
    head = Head.from_cache(cE)
    labels = _i64(cE["cand_index"])[rows]
    assigned_gold = head.cand_token[labels]
    targets = Targets(T=tE, gold_token=assigned_gold, label=labels)
    folds, inner = manifest_fold_plan(inputs.manifest, order)
    fold_sha256 = _digest_folds(folds, inner)
    archived_fold_sha256 = (
        ((archived_metrics.get("checks") or {}).get("folds") or {}).get("sha256")
    )
    if fold_sha256 != archived_fold_sha256:
        raise CorruptionAnalysisError(
            f"fold digest mismatch: recovered={fold_sha256!r}, "
            f"archived={archived_fold_sha256!r}"
        )
    cfg = ProbeConfig(
        q_primary=q,
        n_perm=0,
        backend=archived_backend,
        device=recovery_device,
        dtype="float64",
    )
    recovered = lin_probe_nested_batch(
        S0, [targets], head, cfg, (q,), folds=folds, inner_folds=inner
    )[0][q]
    parity = primary_probe_parity(recovered, archived_metrics["primary_probe"])

    candidates = [str(value) for value in cE["candidates"]]
    if len(candidates) != len(set(candidates)):
        raise CorruptionAnalysisError("candidate strings are not unique")
    if len(set(head.cand_token.tolist())) != len(candidates):
        raise CorruptionAnalysisError(
            "candidate first-token collisions make original-rule labels ambiguous"
        )
    label_validation = validated_corruption_labels(
        order, inputs.manifest, inputs.arm, candidates, labels
    )
    candidate_index = {animal: index for index, animal in enumerate(candidates)}
    missing = sorted(set(label_validation["clean_animals"]) - set(candidate_index))
    if missing:
        raise CorruptionAnalysisError(
            f"clean labels are outside the candidate set: {missing!r}"
        )
    clean_gold = np.asarray(
        [head.cand_token[candidate_index[animal]]
         for animal in label_validation["clean_animals"]],
        dtype=np.int64,
    )
    pred_token = head.tokens(recovered["pred"])
    base_token = head.tokens(recovered["base_pred"])
    rescored = rule_rescore(pred_token, base_token, clean_gold)

    def vector_sha(values: np.ndarray) -> str:
        return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()

    return {
        "assigned_label_as_run": {
            "acc": float(recovered["acc"]),
            "base_acc": float(recovered["base_acc"]),
            "gain": float(recovered["delta"]),
            "micro_acc": float(recovered["micro_acc"]),
            "micro_base_acc": float(recovered["micro_base_acc"]),
            "micro_gain": float(recovered["micro_delta"]),
            "n": int(recovered["n"]),
            "q": int(q),
            "layers": [int(layer) for layer in recovered["layers"]],
            "modal_layer": int(recovered["modal_layer"]),
        },
        "original_rule_rescore": rescored,
        "parity": parity,
        "archived_fold_sha256": fold_sha256,
        "validated_n_changed_subjects": label_validation["n_changed"],
        "validated_realised_changed_fraction": label_validation[
            "realised_changed_fraction"
        ],
        "prediction_candidate_indices_sha256": vector_sha(recovered["pred"]),
        "baseline_candidate_indices_sha256": vector_sha(recovered["base_pred"]),
        "clean_gold_token_vector_sha256": vector_sha(clean_gold),
        "assigned_gold_token_vector_sha256": vector_sha(assigned_gold),
        "same_prediction_vectors_rescored": True,
        "clean_labels_used_in_fit_or_selection": False,
        "archived_compute": {
            "backend": archived_backend,
            "device": archived_device,
            "dtype": archived_dtype,
        },
        "recovery_compute": {
            "backend": archived_backend,
            "device": recovery_device,
            "dtype": "float64",
        },
    }
