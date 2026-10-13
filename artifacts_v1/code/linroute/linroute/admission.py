from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "SCHEMA",
    "GENERATOR",
    "GENERATOR_VERSION",
    "DEFAULTS",
    "HOLDOUT_RULE",
    "HOLDOUT_RULE_VERSION",
    "HOLDOUT_PROTOCOL",
    "V1_HOLDOUT_RULE",
    "V1_PROTOCOL",
    "POOL_EXCLUSION_REASONS",
    "CLASS_CAP_REASON_PREFIX",
    "CLASS_CAP_ORDER_RULE",
    "CLASS_EXCLUSION_REASONS",
    "SUPPORT_RULES",
    "Subject",
    "assert_extends_v1",
    "bound_ab",
    "build_manifest",
    "check_centered_rank",
    "class_cap_reason",
    "class_supports",
    "holdout_ranking",
    "holdout_rule_of",
    "holdout_size",
    "holdout_sizes",
    "inner_partitions",
    "load_records_file",
    "manifest_json",
    "outer_partitions",
    "partitions",
    "per_class_holdout",
    "protocol_of",
    "repair_split",
    "reserve_support",
    "subjects_from_inventory",
    "subjects_from_records",
    "summary_lines",
    "support_counts",
]

SCHEMA = "linroute/admission/2"

SCHEMA_V1 = "linroute/admission/1"

GENERATOR = "scripts/make_split_manifest.py"

GENERATOR_VERSION = "2.0.0"

HOLDOUT_RULE = "mushroom-v2"

HOLDOUT_RULE_VERSION = "2.0.0"

HOLDOUT_PROTOCOL = "heldout-mushroom-v2"

V1_HOLDOUT_RULE = "v1-one-per-class"
V1_PROTOCOL = "v1-one-per-class"

POOL_EXCLUSION_REASONS = (
    "context_inconsistent",
    "no_effective_class",
)

CLASS_CAP_REASON_PREFIX = "class_cap_"
CLASS_EXCLUSION_REASONS = ("support_below_min", "class_cap_78")

SUPPORT_RULES = ("seeded", "protect")

DEFAULTS = {
    "support_rule": "seeded",
    "min_support": 5,
    "min_classes": 5,
    "max_classes": 78,
    "min_rows": 25,
    "q": 16,
    "outer": 10,
    "inner": 5,
    "role_seed": 0,
    "fold_seed": 0,
    "require_context_consistent": True,

    "holdout_divisor": 5,
    "holdout_cap": 10,
    "holdout_min_train": 4,
}

ENTITY_GROUPING = "subject_string"

@dataclass(frozen=True)
class Subject:
    subject_id: str
    subject: str
    effective_class: int

    def as_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "subject": self.subject,
            "effective_class": self.effective_class,
        }

def _coerce(rows: Iterable) -> list[Subject]:
    out: list[Subject] = []
    for row in rows:
        if isinstance(row, Subject):
            out.append(row)
        else:
            sid, subject, label = row
            out.append(Subject(str(sid), str(subject), int(label)))
    return out

def _dedupe(subjects: Sequence[Subject]) -> list[Subject]:
    seen: dict[str, Subject] = {}
    for s in subjects:
        prior = seen.get(s.subject_id)
        if prior is None:
            seen[s.subject_id] = s
        elif prior != s:
            raise ValueError(
                f"subject id {s.subject_id!r} appears twice with different "
                f"content: {prior} vs {s}"
            )
    return sorted(seen.values(), key=lambda s: (s.effective_class, s.subject_id))

def _pool_from(
    rows: Iterable[tuple[str, str, str, object, bool]],
    *,
    require_context_consistent: bool,
) -> tuple[list[Subject], list[dict], dict]:
    pool: list[Subject] = []
    dropped: list[dict] = []
    n_records = 0
    n_included = 0
    for status, sid, subject, label, consistent in rows:
        n_records += 1
        if status != "included":
            continue
        n_included += 1
        if label is None:
            dropped.append({"subject_id": sid, "subject": subject,
                            "reason": "no_effective_class"})
            continue
        if require_context_consistent and not consistent:
            dropped.append({"subject_id": sid, "subject": subject,
                            "reason": "context_inconsistent"})
            continue
        pool.append(Subject(str(sid), str(subject), int(label)))
    counts = {"n_records": n_records, "n_included": n_included}
    dropped.sort(key=lambda d: (d["reason"], d["subject_id"]))
    return _dedupe(pool), dropped, counts

def load_records_file(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def subjects_from_records(
    records: Iterable[dict], *, require_context_consistent: bool = True
) -> tuple[list[Subject], list[dict], dict]:
    return _pool_from(
        (
            (
                r.get("status"),
                r.get("subject_id"),
                r.get("subject"),
                r.get("effective_class"),
                bool(r.get("context_consistent", True)),
            )
            for r in records
        ),
        require_context_consistent=require_context_consistent,
    )

def subjects_from_inventory(
    inv, *, require_context_consistent: bool = True
) -> tuple[list[Subject], list[dict], dict]:
    return _pool_from(
        (
            (
                r.status,
                r.subject_id,
                r.subject,
                r.effective_class,
                bool(r.context_consistent),
            )
            for r in inv.records
        ),
        require_context_consistent=require_context_consistent,
    )

def class_supports(subjects: Sequence[Subject]) -> dict[int, int]:
    supports: dict[int, int] = {}
    for s in subjects:
        supports[s.effective_class] = supports.get(s.effective_class, 0) + 1
    return dict(sorted(supports.items()))

def support_counts(supports: dict[int, int]) -> list[dict]:
    if not supports:
        return []
    top = max(supports.values())
    values = list(supports.values())
    return [{"m": m, "k": sum(1 for v in values if v >= m)}
            for m in range(1, top + 1)]

def _by_class(subjects: Sequence[Subject]) -> dict[int, list[Subject]]:
    groups: dict[int, list[Subject]] = {}
    for s in subjects:
        groups.setdefault(s.effective_class, []).append(s)
    for members in groups.values():
        members.sort(key=lambda s: s.subject_id)
    return dict(sorted(groups.items()))

def reserve_support(
    subjects: Sequence[Subject],
    *,
    n_shots: int,
    rule: str = "seeded",
    seed: int = 0,
) -> tuple[list[Subject], list[Subject]]:
    if rule not in SUPPORT_RULES:
        raise ValueError(f"unknown support rule {rule!r}; expected one of "
                         f"{', '.join(SUPPORT_RULES)}")
    if n_shots < 0:
        raise ValueError(f"n_shots must be >= 0, got {n_shots}")
    pool = _dedupe(subjects)
    if n_shots > len(pool):
        raise ValueError(
            f"cannot reserve {n_shots} support subjects from a pool of "
            f"{len(pool)}"
        )
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])

    if rule == "seeded":
        order = rng.permutation(len(pool)).tolist()
        chosen_idx = set(order[:n_shots])
        support = [pool[i] for i in order[:n_shots]]
        remaining = [pool[i] for i in range(len(pool)) if i not in chosen_idx]
        return _sorted_subjects(support), remaining

    groups = _by_class(pool)

    labels = list(groups)
    priority = {c: int(p) for c, p in zip(labels, rng.permutation(len(labels)))}
    shuffled = {
        c: [groups[c][int(i)] for i in rng.permutation(len(groups[c]))]
        for c in labels
    }
    taken: dict[int, int] = {c: 0 for c in labels}
    support = []
    for _ in range(n_shots):
        c = max(labels, key=lambda c: (len(groups[c]) - taken[c], priority[c]))
        support.append(shuffled[c][taken[c]])
        taken[c] += 1
    reserved = {s.subject_id for s in support}
    remaining = [s for s in pool if s.subject_id not in reserved]
    return _sorted_subjects(support), remaining

def _sorted_subjects(subjects: Iterable[Subject]) -> list[Subject]:
    return sorted(subjects, key=lambda s: (s.effective_class, s.subject_id))

def bound_ab(n_rows: int, outer: int, inner: int) -> tuple[int, int]:
    a = n_rows - math.ceil(n_rows / outer)
    b = a - math.ceil(a / inner)
    return a, b

def class_cap_reason(max_classes: int) -> str:
    return f"{CLASS_CAP_REASON_PREFIX}{max_classes}"

CLASS_CAP_ORDER_RULE = (
    "class ids in draw order; the cap keeps the first max_classes of them"
)

def _cap_classes(
    qualifying: Sequence[int], *, max_classes: int, seed: int,
    order: Sequence[int] | None = None,
) -> tuple[list[int], list[int], list[int], str]:
    classes = sorted(qualifying)
    if order is None:
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
        draw = [classes[i] for i in rng.permutation(len(classes)).tolist()]
        source = "derived"
    else:
        draw = [int(c) for c in order]
        if sorted(draw) != classes:
            missing = sorted(set(classes) - set(draw))
            extra = sorted(set(draw) - set(classes))
            raise ValueError(
                f"the reused class-cap order is not a permutation of the "
                f"{len(classes)} qualifying classes: it lists {len(draw)} ids, "
                f"{len(missing)} qualifying class(es) are absent from it "
                f"{missing[:8]}, {len(extra)} id(s) in it do not qualify "
                f"{extra[:8]}"
                + (f", and it repeats {len(draw) - len(set(draw))} id(s)"
                   if len(set(draw)) != len(draw) else "")
                + ".  Reusing an order across two different pools is "
                "meaningless; rebuild without --reuse-class-cap-order."
            )
        source = "reused"
    kept = sorted(draw[:max_classes])
    dropped = sorted(draw[max_classes:])
    return kept, dropped, draw, source

def _admit(
    supports: dict[int, int],
    *,
    min_support: int,
    min_classes: int,
    max_classes: int,
    min_rows: int,
    role_seed: int,
    class_cap_order: Sequence[int] | None = None,
    holdout_divisor: int = DEFAULTS["holdout_divisor"],
    holdout_cap: int = DEFAULTS["holdout_cap"],
    holdout_min_train: int = DEFAULTS["holdout_min_train"],
) -> tuple[list[int], list[dict], int, list[str], dict]:
    qualifying = [c for c, s in supports.items() if s >= min_support]
    retained, capped, cap_order, cap_order_source = _cap_classes(
        qualifying, max_classes=max_classes, seed=role_seed,
        order=class_cap_order,
    )
    capped_set = set(capped)
    excluded = [
        {"effective_class": c, "support": supports[c],
         "reason": (class_cap_reason(max_classes) if c in capped_set
                    else "support_below_min")}
        for c in supports
        if c in capped_set or supports[c] < min_support
    ]
    h = holdout_sizes(
        supports, retained, divisor=holdout_divisor, cap=holdout_cap,
        min_train=holdout_min_train,
    )
    n_rows = sum(supports[c] - h[c] for c in retained)
    failing = []
    if len(retained) < min_classes:
        failing.append("min_classes")
    if n_rows < min_rows:
        failing.append("min_rows")
    class_cap = {
        "max_classes": max_classes,
        "applied": bool(capped),
        "n_qualifying": len(qualifying),
        "n_retained": len(retained),
        "n_excluded": len(capped),
        "seed": role_seed,
        "reason": class_cap_reason(max_classes),

        "order": cap_order,
        "order_source": cap_order_source,
        "order_rule": CLASS_CAP_ORDER_RULE,
    }
    return retained, excluded, n_rows, failing, class_cap

def holdout_size(
    n_c: int,
    *,
    divisor: int = DEFAULTS["holdout_divisor"],
    cap: int = DEFAULTS["holdout_cap"],
    min_train: int = DEFAULTS["holdout_min_train"],
) -> int:
    n_c = int(n_c)
    if divisor < 1:
        raise ValueError(f"holdout divisor must be >= 1, got {divisor}")
    if cap < 1:
        raise ValueError(f"holdout cap must be >= 1, got {cap}")
    if min_train < 0:
        raise ValueError(f"holdout min_train must be >= 0, got {min_train}")
    return max(1, min(cap, max(1, n_c // divisor), n_c - min_train))

def holdout_sizes(
    supports: dict[int, int],
    retained: Sequence[int],
    *,
    divisor: int = DEFAULTS["holdout_divisor"],
    cap: int = DEFAULTS["holdout_cap"],
    min_train: int = DEFAULTS["holdout_min_train"],
) -> dict[int, int]:
    return {
        int(c): holdout_size(supports[c], divisor=divisor, cap=cap,
                             min_train=min_train)
        for c in sorted(retained)
    }

def holdout_ranking(
    subjects: Sequence[Subject], retained: Sequence[int], *, seed: int
) -> dict[int, list[Subject]]:
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    groups = _by_class(subjects)
    order = [c for c in retained if groups.get(c)]

    ranked: dict[int, list[Subject]] = {}
    for c in order:
        members = groups[c]
        ranked[c] = [members[int(rng.integers(len(members)))]]
    for c in order:
        members = groups[c]
        first = ranked[c][0].subject_id
        rest = [m for m in members if m.subject_id != first]
        perm = rng.permutation(len(rest)).tolist()
        ranked[c].extend(rest[int(i)] for i in perm)
    return ranked

def _roles_from(
    ranking: dict[int, list[Subject]],
    subjects: Sequence[Subject],
    sizes: dict[int, int],
) -> tuple[list[Subject], list[Subject]]:
    held = [s for c, members in ranking.items()
            for s in members[: sizes.get(c, 0)]]
    chosen = {s.subject_id for s in held}
    retained = set(ranking)
    train = [s for s in subjects
             if s.effective_class in retained and s.subject_id not in chosen]
    return _sorted_subjects(held), _sorted_subjects(train)

def _fold_view(train: Sequence[Subject], folds: list[dict], *,
               outer: int, inner: int) -> dict:
    return {
        "train": {"subjects": [s.as_dict() for s in train]},
        "folds": {"outer": folds},
        "parameters": {"outer": outer, "inner": inner},
    }

def _attempt_split(
    train: Sequence[Subject],
    retained: Sequence[int],
    *,
    outer: int,
    inner: int,
    q: int,
    fold_seed: int,
    min_rows: int,
) -> tuple[list[dict], dict, bool, str | None]:
    buildable = len(train) >= outer and len(retained) > 0
    folds = (_nested_folds(train, outer=outer, inner=inner, seed=fold_seed)
             if buildable else [])
    checks = _validate(_fold_view(train, folds, outer=outer, inner=inner),
                       train, retained, q=q)
    if not buildable:
        checks["failures"].append({
            "partition": None, "check": "folds_not_built",
            "n_rows": len(train), "required": outer,
        })
        checks["n_failures"] = len(checks["failures"])
    if len(train) < min_rows:
        return folds, checks, False, "min_rows"
    if checks["failures"]:
        return folds, checks, False, str(checks["failures"][0]["check"])
    return folds, checks, True, None

def repair_split(
    ranking: dict[int, list[Subject]],
    subjects: Sequence[Subject],
    retained: Sequence[int],
    sizes: dict[int, int],
    *,
    outer: int,
    inner: int,
    q: int,
    fold_seed: int,
    min_rows: int,
) -> dict:
    sizes = {int(c): int(n) for c, n in sizes.items()}
    ledger: list[dict] = []
    depth = 0
    while True:
        held, train = _roles_from(ranking, subjects, sizes)
        folds, checks, ok, failing = _attempt_split(
            train, retained, outer=outer, inner=inner, q=q,
            fold_seed=fold_seed, min_rows=min_rows,
        )
        if ok:
            break
        movable = [c for c in sorted(ranking) if sizes.get(c, 0) > 1]
        if not movable:
            break

        c = max(movable, key=lambda c: sizes[c])
        sizes[c] -= 1
        returned = ranking[c][sizes[c]]
        ledger.append({
            "subject_id": returned.subject_id,
            "subject": returned.subject,
            "effective_class": int(c),
            "rank": int(sizes[c]),
            "reason": failing,
        })
        depth += 1
    return {
        "holdout": held,
        "train": train,
        "sizes": sizes,
        "folds": folds,
        "checks": checks,
        "feasible": ok,
        "failing_check": failing,
        "returned_to_train": ledger,
        "fallback_applied": bool(ledger),
        "fallback_depth": depth,
        "fallback_exhausted": (not ok) and bool(
            not [c for c in ranking if sizes.get(c, 0) > 1]
        ),
    }

def per_class_holdout(manifest: dict) -> dict[int, int]:
    per_class = (manifest.get("holdout") or {}).get("per_class")
    if isinstance(per_class, dict) and per_class:
        return {int(c): int(v["h_final"]) for c, v in per_class.items()}
    counts: dict[int, int] = {}
    for s in (manifest.get("holdout") or {}).get("subjects", []):
        c = int(s["effective_class"])
        counts[c] = counts.get(c, 0) + 1
    return dict(sorted(counts.items()))

def holdout_rule_of(manifest: dict | None) -> str | None:
    if not manifest:
        return None
    rule = (manifest.get("holdout") or {}).get("rule")
    if rule:
        return str(rule)
    if manifest.get("schema") == SCHEMA_V1:
        return V1_HOLDOUT_RULE
    return None

def protocol_of(holdout_rule: str | None) -> str | None:
    if holdout_rule is None:
        return None
    if holdout_rule == HOLDOUT_RULE:
        return HOLDOUT_PROTOCOL
    if holdout_rule == V1_HOLDOUT_RULE:
        return V1_PROTOCOL
    return str(holdout_rule)

def assert_extends_v1(manifest: dict, v1_manifest: dict) -> dict:
    v1_ids = {s["subject_id"] for s in v1_manifest["holdout"]["subjects"]}
    rank0 = {s["subject_id"] for s in manifest["holdout"]["subjects"]
             if int(s.get("rank", 0)) == 0}
    subjects = {s["subject_id"] for s in manifest["holdout"]["subjects"]}
    missing = sorted(v1_ids - subjects)
    if missing:
        raise ValueError(
            f"{len(missing)} v1 held-out subjects are absent from the v2 "
            f"held-out set, e.g. {missing[:4]}: the seeded order was not "
            f"extended, it was redrawn"
        )
    moved = sorted(v1_ids ^ rank0)
    if moved:
        raise ValueError(
            f"the v2 rank-0 held-out subjects are not the v1 held-out set: "
            f"{moved[:4]} differ"
        )
    return {
        "v1_schema": v1_manifest.get("schema"),
        "v1_n_holdout": len(v1_ids),
        "contains_v1_holdout": True,
        "v1_holdout_is_rank_zero": True,
    }

def _deal(
    members_by_class: dict[int, list[str]], n_folds: int, rng
) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(n_folds)]
    cursor = 0
    for c in sorted(members_by_class):
        ids = list(members_by_class[c])
        order = rng.permutation(len(ids))
        for j, k in enumerate(order):
            buckets[(cursor + j) % n_folds].append(ids[int(k)])
        cursor = (cursor + len(ids)) % n_folds
    return [sorted(b) for b in buckets]

def _group_ids(ids: Sequence[str], label: dict[str, int]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    for sid in ids:
        groups.setdefault(label[sid], []).append(sid)
    for members in groups.values():
        members.sort()
    return groups

def _nested_folds(
    train: Sequence[Subject], *, outer: int, inner: int, seed: int
) -> list[dict]:
    label = {s.subject_id: s.effective_class for s in train}
    ids = [s.subject_id for s in train]
    root = np.random.SeedSequence(seed)
    outer_ss, inner_parent = root.spawn(2)
    outer_folds = _deal(_group_ids(ids, label), outer,
                        np.random.default_rng(outer_ss))
    inner_seeds = inner_parent.spawn(len(outer_folds))

    out = []
    for o, fold in enumerate(outer_folds):
        held = set(fold)
        fitting = [sid for sid in ids if sid not in held]
        inner_folds = _deal(_group_ids(fitting, label), inner,
                            np.random.default_rng(inner_seeds[o]))
        out.append({
            "index": o,
            "validation": sorted(fold),
            "n_fitting": len(fitting),
            "inner": [
                {"index": i, "validation": sorted(f), "n_fitting": len(fitting) - len(f)}
                for i, f in enumerate(inner_folds)
            ],
        })
    return out

def partitions(manifest: dict) -> Iterator[tuple[str, list[str], list[str]]]:
    yield from outer_partitions(manifest)
    yield from inner_partitions(manifest)

def outer_partitions(manifest: dict) -> Iterator[tuple[str, list[str], list[str]]]:
    ids = [s["subject_id"] for s in manifest["train"]["subjects"]]
    for fold in manifest["folds"]["outer"]:
        held = set(fold["validation"])
        yield (f"outer/{fold['index']}",
               [sid for sid in ids if sid not in held],
               list(fold["validation"]))

def inner_partitions(manifest: dict) -> Iterator[tuple[str, list[str], list[str]]]:
    ids = [s["subject_id"] for s in manifest["train"]["subjects"]]
    for fold in manifest["folds"]["outer"]:
        held = set(fold["validation"])
        outer_fit = [sid for sid in ids if sid not in held]
        for sub in fold["inner"]:
            inner_held = set(sub["validation"])
            yield (f"outer/{fold['index']}/inner/{sub['index']}",
                   [sid for sid in outer_fit if sid not in inner_held],
                   list(sub["validation"]))

def _validate(
    manifest: dict, train: Sequence[Subject], retained: Sequence[int], *, q: int
) -> dict:
    label = {s.subject_id: s.effective_class for s in train}
    wanted = set(retained)
    failures: list[dict] = []
    outer_rows: list[int] = []
    inner_rows: list[int] = []

    for name, fitting, validation in partitions(manifest):
        (inner_rows if "/inner/" in name else outer_rows).append(len(fitting))
        if len(fitting) < q + 1:
            failures.append({
                "partition": name, "check": "fitting_rows",
                "n_rows": len(fitting), "required": q + 1,
            })
        missing = sorted(wanted - {label[sid] for sid in fitting})
        if missing:
            failures.append({
                "partition": name, "check": "class_present",
                "missing_classes": missing,
            })
        if not validation:
            failures.append({"partition": name, "check": "validation_nonempty"})

    n_rows = len(train)
    a, b = bound_ab(n_rows, manifest["parameters"]["outer"],
                    manifest["parameters"]["inner"]) if n_rows else (0, 0)
    return {
        "q": q,
        "min_fitting_rows": q + 1,
        "entity_grouping": ENTITY_GROUPING,
        "n_outer_partitions": len(outer_rows),
        "n_inner_partitions": len(inner_rows),
        "min_outer_fitting_rows": min(outer_rows) if outer_rows else None,
        "min_inner_fitting_rows": min(inner_rows) if inner_rows else None,
        "bound": {
            "n_rows": n_rows, "a": a, "b": b, "b_meets_q": b >= q + 1,
            "note": "a = N - ceil(N/outer), b = a - ceil(a/inner); centering "
                    "caps the rank of a fitting partition at b - 1",
        },
        "n_failures": len(failures),
        "failures": failures,
    }

def build_manifest(
    pool: Sequence,
    *,
    relation: str,
    tokenizer: str,
    source: dict | None = None,
    n_shots: int | None = None,
    support_rule: str = DEFAULTS["support_rule"],
    min_support: int = DEFAULTS["min_support"],
    min_classes: int = DEFAULTS["min_classes"],
    max_classes: int = DEFAULTS["max_classes"],
    min_rows: int = DEFAULTS["min_rows"],
    q: int = DEFAULTS["q"],
    outer: int = DEFAULTS["outer"],
    inner: int = DEFAULTS["inner"],
    role_seed: int = DEFAULTS["role_seed"],
    fold_seed: int = DEFAULTS["fold_seed"],
    require_context_consistent: bool = DEFAULTS["require_context_consistent"],
    holdout_divisor: int = DEFAULTS["holdout_divisor"],
    holdout_cap: int = DEFAULTS["holdout_cap"],
    holdout_min_train: int = DEFAULTS["holdout_min_train"],
    class_cap_order: Sequence[int] | None = None,
    dropped: Sequence[dict] = (),
    record_counts: dict | None = None,
    notes: Sequence[str] = (),
    vetoes: Sequence[str] = (),
    git_commit: str | None = None,
    v1_manifest: dict | None = None,
    v1_manifest_sha256: str | None = None,
) -> dict:
    if n_shots is None:
        from linroute import prompts
        n_shots = prompts.few_shot_spec().n_shots
    if outer < 2 or inner < 2:
        raise ValueError("outer and inner fold counts must both be >= 2")

    eligible = _dedupe(_coerce(pool))
    supports_pre = class_supports(eligible)
    support, remaining = reserve_support(
        eligible, n_shots=n_shots, rule=support_rule, seed=role_seed
    )
    supports = class_supports(remaining)
    retained, excluded, n_rows_target, failing, class_cap = _admit(
        supports, min_support=min_support, min_classes=min_classes,
        max_classes=max_classes, min_rows=min_rows, role_seed=role_seed,
        class_cap_order=class_cap_order,
        holdout_divisor=holdout_divisor, holdout_cap=holdout_cap,
        holdout_min_train=holdout_min_train,
    )

    h_target = holdout_sizes(
        supports, retained, divisor=holdout_divisor, cap=holdout_cap,
        min_train=holdout_min_train,
    )
    ranking = holdout_ranking(remaining, retained, seed=role_seed)
    split = repair_split(
        ranking, remaining, retained, h_target,
        outer=outer, inner=inner, q=q, fold_seed=fold_seed, min_rows=min_rows,
    )
    held, train = split["holdout"], split["train"]
    h_final = split["sizes"]
    n_rows = len(train)

    failing = [f for f in failing if f != "min_rows"]
    if n_rows < min_rows:
        failing.append("min_rows")
    if not split["feasible"] and split["failing_check"] != "min_rows":
        failing.append("fold_checks")
    failing = failing + [v for v in vetoes if v not in failing]
    admitted = not failing

    rank_of = {s.subject_id: r
               for c, members in ranking.items()
               for r, s in enumerate(members)}

    reserved_per_class: dict[int, int] = {}
    for s in support:
        reserved_per_class[s.effective_class] = (
            reserved_per_class.get(s.effective_class, 0) + 1
        )
    excluded_reason = {e["effective_class"]: e["reason"] for e in excluded}
    retained_set = set(retained)
    classes = [
        {
            "effective_class": c,
            "support_pre_reservation": supports_pre.get(c, 0),
            "support": supports.get(c, 0),
            "n_reserved_for_support": reserved_per_class.get(c, 0),
            "retained": c in retained_set,
            "reason": (None if c in retained_set
                       else excluded_reason.get(c, "support_below_min")),
        }
        for c in sorted(set(supports_pre) | set(supports))
    ]

    if git_commit is None:
        from linroute.provenance import git_commit as _git_commit
        git_commit = _git_commit()

    manifest: dict = {
        "schema": SCHEMA,
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "git_commit": git_commit,
        "relation": relation,
        "tokenizer": tokenizer,
        "entity_grouping": ENTITY_GROUPING,
        "source": dict(source) if source else {"kind": "in_memory_inventory",
                                               "description": "in-memory inventory"},
        "parameters": {
            "n_shots": n_shots,
            "support_rule": support_rule,
            "min_support": min_support,
            "min_classes": min_classes,
            "max_classes": max_classes,
            "min_rows": min_rows,
            "q": q,
            "outer": outer,
            "inner": inner,
            "role_seed": role_seed,
            "fold_seed": fold_seed,
            "require_context_consistent": require_context_consistent,
            "offset_rule": "cumulative_class_size",

            "holdout_rule": HOLDOUT_RULE,
            "holdout_divisor": holdout_divisor,
            "holdout_cap": holdout_cap,
            "holdout_min_train": holdout_min_train,
        },
        "pool": {
            "n_eligible": len(eligible),
            "n_dropped": len(dropped),
            "dropped": [dict(d) for d in dropped],
            **(dict(record_counts) if record_counts else {}),
        },
        "support": {
            "rule": support_rule,
            "n_shots": n_shots,
            "subjects": [s.as_dict() for s in support],
        },
        "classes": classes,
        "support_counts": support_counts(supports),
        "admission": {
            "admitted": admitted,
            "failing_conditions": failing,
            "vetoes": list(vetoes),
            "class_cap": class_cap,
            "n_classes_pool": len(supports_pre),
            "n_classes_retained": len(retained),
            "n_classes_excluded": len(excluded),
            "excluded_classes": excluded,
            "n_rows": n_rows,
            "n_rows_target": n_rows_target,
            "n_held_out": len(held),
        },
        "holdout": {
            "n": len(held),
            "rule": HOLDOUT_RULE,
            "rule_version": HOLDOUT_RULE_VERSION,
            "protocol": HOLDOUT_PROTOCOL,
            "formula": "h_c = min(cap, max(1, n_c // divisor), n_c - min_train)",
            "parameters": {
                "divisor": holdout_divisor,
                "cap": holdout_cap,
                "min_train_per_class": holdout_min_train,
            },
            "per_class": {
                str(c): {
                    "support": supports[c],
                    "h_target": h_target[c],
                    "h_final": h_final[c],
                }
                for c in retained
            },
            "returned_to_train": split["returned_to_train"],
            "fallback_applied": split["fallback_applied"],
            "fallback_depth": split["fallback_depth"],
            "fallback_exhausted": split["fallback_exhausted"],

            "extends": ({"manifest_sha256": v1_manifest_sha256}
                        if (v1_manifest is not None or v1_manifest_sha256)
                        else None),

            "subjects": [{**s.as_dict(), "rank": rank_of[s.subject_id]}
                         for s in held],
        },
        "train": {"n": len(train), "subjects": [s.as_dict() for s in train]},
        "centered_rank_checked": False,
        "notes": list(notes),
    }

    manifest["folds"] = {
        "outer": split["folds"],
        "fitting_is_complement": True,
    }
    checks = split["checks"]
    checks["admitted"] = admitted
    checks["passed"] = admitted and not checks["failures"]
    checks["holdout_rule"] = HOLDOUT_RULE
    checks["fallback_applied"] = split["fallback_applied"]
    manifest["checks"] = checks
    if v1_manifest is not None:
        manifest["holdout"]["extends"] = {
            "manifest_sha256": v1_manifest_sha256,
            **assert_extends_v1(manifest, v1_manifest),
        }
    return manifest

def manifest_json(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

def _holdout_line(manifest: dict) -> str:
    adm = manifest["admission"]
    block = manifest["holdout"]
    params = block["parameters"]
    values = sorted(v["h_final"] for v in block["per_class"].values())
    if values:
        mid = values[len(values) // 2]
        spread = (f"h_c from {values[0]} to {values[-1]}, median {mid}"
                  if values[0] != values[-1] else f"h_c = {values[0]} in every class")
    else:
        spread = "no retained class"
    return (
        f"|H| = {adm['n_held_out']} under the {block['rule']} rule "
        f"({block['formula']}, divisor {params['divisor']}, cap "
        f"{params['cap']}, min_train {params['min_train_per_class']}): "
        f"{spread}.  N = |T| = {adm['n_rows']}."
    )

def _fallback_line(manifest: dict) -> str:
    block = manifest["holdout"]
    if not block["fallback_applied"]:
        return ("The fallback did not fire: the first split passed the row and "
                "fold checks.")
    reasons = sorted({r["reason"] for r in block["returned_to_train"] if r["reason"]})
    tail = (" and the comparison is still infeasible"
            if block["fallback_exhausted"] else "")
    return (
        f"The fallback returned {block['fallback_depth']} held-out subject(s) "
        f"to T, largest h_c first and last-drawn first, for "
        f"{', '.join(reasons) or 'the fold checks'}{tail}."
    )

def summary_lines(manifest: dict) -> list[str]:
    adm = manifest["admission"]
    cap = adm["class_cap"]
    checks = manifest["checks"]
    bound = checks["bound"]
    pool = manifest["pool"]
    lines = [
        f"{manifest['relation']} under {manifest['tokenizer']}: "
        f"{pool['n_eligible']} eligible subjects"
        + (f", {pool['n_dropped']} dropped by this module" if pool["n_dropped"] else "")
        + ".",
        f"Reserved {manifest['support']['n_shots']} five-shot support subjects "
        f"first, by the {manifest['support']['rule']!r} rule "
        f"(role seed {manifest['parameters']['role_seed']}).",
        f"After reservation {adm['n_classes_pool']} classes; "
        f"{adm['n_classes_retained']} retained at support "
        f">= {manifest['parameters']['min_support']}, "
        f"{adm['n_classes_excluded']} excluded.",
        _holdout_line(manifest),
        _fallback_line(manifest),
        (f"The {cap['max_classes']}-class cap applied: "
         f"{cap['n_qualifying']} classes qualified on support and "
         f"{cap['n_retained']} were drawn uniformly at role seed "
         f"{cap['seed']}, the other {cap['n_excluded']} excluded as "
         f"{cap['reason']}."
         if cap["applied"] else
         f"The {cap['max_classes']}-class cap did not apply: "
         f"{cap['n_qualifying']} classes qualified on support."),
        f"Admitted: {'yes' if adm['admitted'] else 'no'}"
        + (f" (failing: {', '.join(adm['failing_conditions'])})"
           if adm["failing_conditions"] else "")
        + ".",
        f"Folds: {manifest['parameters']['outer']} outer x "
        f"{manifest['parameters']['inner']} inner, fold seed "
        f"{manifest['parameters']['fold_seed']}; smallest outer fitting "
        f"partition {checks['min_outer_fitting_rows']} rows, smallest inner "
        f"{checks['min_inner_fitting_rows']}, against the closed-form bound "
        f"a = {bound['a']}, b = {bound['b']} and the q = {checks['q']} "
        f"requirement of {checks['min_fitting_rows']} rows.",
        f"Checks: {'passed' if checks['passed'] else 'FAILED'}"
        + (f", {checks['n_failures']} failing partition checks"
           if checks["n_failures"] else "")
        + ".",
        "Centered numerical rank is state-dependent and was not checked here; "
        "run linroute.admission.check_centered_rank with the cached states.",
    ]
    for failure in checks["failures"][:10]:
        lines.append(f"  failed: {failure['partition']} {failure['check']}")
    if checks["n_failures"] > 10:
        lines.append(f"  ... and {checks['n_failures'] - 10} more")
    return lines

def check_centered_rank(states: np.ndarray, manifest: dict, q: int = 16) -> dict:
    from linroute.probes import feasible_q

    states = np.asarray(states, dtype=np.float64)
    order = [s["subject_id"] for s in manifest["train"]["subjects"]]
    if states.ndim != 2:
        raise ValueError(f"states must be 2-D (n_T, d), got shape {states.shape}")
    if states.shape[0] != len(order):
        raise ValueError(
            f"states has {states.shape[0]} rows but the manifest's T has "
            f"{len(order)} subjects"
        )
    index = {sid: i for i, sid in enumerate(order)}

    results = []
    for name, fitting, _validation in partitions(manifest):
        rows = states[[index[sid] for sid in fitting]]
        n, d = rows.shape
        if n == 0:
            results.append({"partition": name, "n_rows": 0, "rank": 0,
                            "feasible": False})
            continue
        centered = rows - rows.mean(0, keepdims=True)
        sv = np.linalg.svd(centered, compute_uv=False)
        tol = sv[0] * max(n, d) * np.finfo(np.float64).eps if sv[0] > 0 else 0.0
        rank = int((sv > tol).sum()) if sv[0] > 0 else 0
        results.append({
            "partition": name, "n_rows": n, "rank": rank,
            "feasible": bool(feasible_q(sv, n, d, q)),
        })

    infeasible = [r["partition"] for r in results if not r["feasible"]]
    return {
        "q": q,
        "n_partitions": len(results),
        "n_infeasible": len(infeasible),
        "infeasible": infeasible,
        "passed": not infeasible,
        "partitions": results,
    }
