#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linroute import transfer
from linroute.provenance import git_commit

MODELS = ("OLMo-3-1025-7B", "Qwen3.5-9B")

SCREEN_ALIASES = {"Qwen3.8-27B": "Qwen3.5-9B"}
REPEATS = (0, 1, 2)

NAME_KEYS = ("relation_slug", "relation", "slug", "name")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", required=True,
                   help="directory searched recursively for metrics.json files")
    p.add_argument("--index", required=True, help="results-index.json to write")
    p.add_argument("--rows", required=True, help="rows.jsonl to write")
    p.add_argument("--relations", default=None,
                   help="the admission screen JSON (relations and their verdicts)")
    p.add_argument("--models", nargs="+", default=None,
                   help="default: every model the --relations screen covers, or "
                        f"{list(MODELS)} when no screen is given")
    p.add_argument("--repeats", nargs="+", type=int, default=list(REPEATS))
    p.add_argument("--manifests", default=None,
                   help="directory the index's manifest paths are given relative to")
    return p.parse_args(argv)

def models_of(screen: Sequence[dict], rows: Sequence[dict]) -> list[str]:
    out: list[str] = []
    for source in (screen, rows):
        for row in source:
            name = row.get("model")
            if name and str(name) not in out:
                out.append(str(name))
    return out or list(MODELS)

def _json(path: Path) -> dict | list:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_screen(path: Path | None) -> list[dict]:
    if path is None:
        return []
    data = _json(path)
    if isinstance(data, dict):
        for key in ("rows", "relations", "screen"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise SystemExit(
                f"{path}: expected a top-level 'rows' list of (relation, model) "
                f"screen rows; got keys {sorted(data)[:8]}"
            )
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a list of screen rows, got {type(data).__name__}")
    return [dict(row) for row in data]

def load_agreement(path: Path | None, screen: Sequence[dict]) -> dict:
    if path is None:
        return {}
    data = _json(path)
    if not isinstance(data, dict):
        return {}
    agreement = data.get("agreement")
    if not isinstance(agreement, dict):
        return {}
    slug_of = {}
    for row in screen:
        human = row.get("relation")
        slug = row.get("relation_slug")
        if human and slug:
            slug_of[str(human)] = _run_slug(row, str(slug))
    out = {}
    for human, block in agreement.items():
        if isinstance(block, dict):
            out[slug_of.get(str(human), str(human))] = block
    return out

def _partition_flags(block: dict | None) -> dict:
    if not block:
        return {"partition_agrees_across_models": None,
                "model_specific_partition": None}
    agrees = block.get("agree_up_to_class_names")
    return {
        "partition_agrees_across_models": agrees,
        "model_specific_partition": None if agrees is None else (not agrees),
    }

def _name(row: dict) -> str:
    for key in NAME_KEYS:
        if row.get(key):
            return _run_slug(row, str(row[key]))
    raise SystemExit(f"a screen entry has no relation name: keys {sorted(row)[:8]}")

def _run_slug(row: dict, name: str) -> str:
    manifest = row.get("manifest")
    if manifest:
        stem = Path(str(manifest)).stem
        if stem != name and stem.casefold() == name.casefold():
            return stem
    return name

def _admitted(row: dict) -> bool:
    if "admitted" in row:
        return bool(row["admitted"])
    return str(row.get("status", "")).lower() in ("admitted", "complete", "in_progress")

def _reason(row: dict) -> str | None:
    reasons = row.get("reasons")
    if isinstance(reasons, (list, tuple)) and reasons:
        return "; ".join(str(r) for r in reasons)
    if isinstance(reasons, str) and reasons:
        return reasons
    return row.get("reason")

def _counts(row: dict) -> dict:
    manifest = row.get("manifest")
    if isinstance(manifest, dict):
        manifest_path = manifest.get("path")
        manifest_sha = manifest.get("sha256")
    else:
        manifest_path = manifest or row.get("manifest_path")
        manifest_sha = row.get("manifest_sha256")
    return {
        "relation_name": row.get("relation"),
        "n_classes": row.get("n_classes"),
        "n_train": row.get("n_train", row.get("N")),
        "n_heldout": row.get("n_heldout"),
        "holdout_rule": row.get("holdout_rule"),
        "holdout_profile": row.get("holdout_profile"),
        "n_support": row.get("n_support"),
        "support_profile": row.get("support_profile"),
        "K": row.get("K"),
        "prefix_uniform": row.get("prefix_uniform"),
        "checks_passed": row.get("checks_passed"),
        "min_inner_fitting_rows": row.get("min_inner_fitting_rows"),
        "vetoes": row.get("vetoes"),
        "manifest": manifest_path,
        "manifest_sha256": manifest_sha,
    }

def screen_alias_of(model: str | None, screen: Sequence[dict]) -> str | None:
    own = {str(r["model"]) for r in screen if r.get("model")}
    lender = SCREEN_ALIASES.get(str(model))
    if model is None or str(model) in own or lender is None or lender not in own:
        return None
    return lender

def screen_index(screen: Sequence[dict], models: Sequence[str]) -> dict:
    out: dict[str, dict[str, dict]] = {}
    for row in screen:
        name = _name(row)
        target = out.setdefault(name, {})
        model = row.get("model")
        if model:
            target[str(model)] = {**row, "admission_scope": "model"}
        else:
            for m in models:
                target.setdefault(m, {**row, "model": m,
                                      "admission_scope": "all_models"})
    for name, target in out.items():
        for m in models:
            lender = screen_alias_of(m, screen)
            if lender is None or str(m) in target or lender not in target:
                continue
            target[str(m)] = {**target[lender], "model": str(m),
                              "admission_scope": "alias",
                              "screen_alias_of": lender}
    return out

def find_metrics(root: Path) -> list[Path]:
    return sorted(root.rglob("metrics.json"))

def eligibility_rejection(payload: dict) -> str | None:
    status = payload.get("status") or "complete"
    if status == "failed" or payload.get("error"):
        return None

    if status != "complete" and not payload.get("core"):
        return None

    rule = payload.get("holdout_rule")
    want_rule = transfer.REQUIRED_HOLDOUT_RULE
    if rule != want_rule:
        if rule is None:
            return (
                f"protocol v1: this payload carries no holdout_rule, so it was "
                f"produced before the {want_rule} protocol existed.  Its "
                f"held-out set is one subject per class and its T is a "
                f"different fine-tuning set; the row is kept as v1 evidence and "
                f"never pooled with {want_rule}."
            )
        return (
            f"protocol {transfer.admission.protocol_of(rule)}: holdout_rule is "
            f"{rule!r}, not the required {want_rule!r}; the row is kept and "
            f"never pooled."
        )

    missing = [k for k in transfer.ELIGIBILITY_FIELDS
               if payload.get(k) is None]
    if missing:
        return (
            "missing eligibility fields: " + ", ".join(missing)
            + ".  A payload that claims a production row must carry the facts "
              "that decide whether it is one."
        )

    horizon = transfer.PRODUCTION_EPOCHS
    reasons: list[str] = []

    def as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    declared = as_int(payload.get("production_horizon"))
    if declared != horizon:
        reasons.append(
            f"production_horizon {payload.get('production_horizon')!r} is not "
            f"the frozen {horizon}"
        )
    for key in ("planned_epochs", "epochs_done", "final_epoch"):
        value = as_int(payload.get(key))
        if value != horizon:
            reasons.append(f"{key} {payload.get(key)!r} != production {horizon}")

    limit = as_int(payload.get("limit_train"))
    if limit is None:
        reasons.append(f"limit_train {payload.get('limit_train')!r} is not a number")
    elif limit:
        reasons.append(f"limit_train {limit} truncates the training set")
    records = as_int(payload.get("n_train_records"))
    exposed = as_int(payload.get("n_exposed_per_epoch"))
    if records is None or records <= 0:
        reasons.append(f"n_train_records {payload.get('n_train_records')!r} is not a count")
    elif exposed is None:
        reasons.append(
            f"n_exposed_per_epoch {payload.get('n_exposed_per_epoch')!r} is not a count"
        )
    elif exposed != records:
        reasons.append(
            f"{exposed} training records exposed per epoch != the arm's {records}"
        )
    if not reasons:
        return None
    return (
        "claims a production row it is not: " + "; ".join(reasons)
    )

def _run_entry(path: Path, payload: dict, root: Path) -> dict:
    status = payload.get("status") or ("failed" if payload.get("error") else "complete")
    rejection = eligibility_rejection(payload)
    if rejection:
        status = "rejected"
    return {
        "run_id": payload.get("run_id") or transfer.run_id(
            payload.get("model", "?"), payload.get("relation", "?"),
            payload.get("repeat", 0), payload.get("rho"),
        ),
        "model": payload.get("model"),
        "relation": payload.get("relation"),
        "repeat": payload.get("repeat"),
        "status": status,
        "reason": rejection or payload.get("error"),
        "host": payload.get("host"),
        "started": payload.get("started"),
        "finished": payload.get("finished"),
        "epochs_done": payload.get("epochs_done"),
        "final_epoch": payload.get("final_epoch"),
        "metrics_file": str(path.relative_to(root)) if root in path.parents else str(path),
        "log": payload.get("log"),
    }

def _row_of(payload: dict, commit: str) -> dict:
    if payload.get("status") == "failed" or payload.get("error"):
        return transfer.failed_row(
            model_name=payload.get("model", "?"),
            relation=payload.get("relation", "?"),
            repeat=int(payload.get("repeat") or 0),
            git_commit=payload.get("git_commit") or commit,
            error=payload.get("error") or "unknown failure",
            holdout_rule=payload.get("holdout_rule"),
            rho=payload.get("rho"),
        )
    rejection = eligibility_rejection(payload)
    if rejection:
        row = transfer.flat_row(
            payload, git_commit=payload.get("git_commit") or commit,
            status="rejected",
        )
        row["core"] = False
        row["error"] = rejection
        return row
    return transfer.flat_row(payload, git_commit=payload.get("git_commit") or commit)

def _relation_status(
    admitted: bool | None, reason: str | None, repeats: dict
) -> tuple[str, str | None]:
    if admitted is None:
        return "unscreened", reason or "no screen row for this (relation, model)"
    if not admitted:
        return "rejected", reason or "not admitted by the screen"
    values = set(repeats.values())
    if "rejected" in values:
        return "failed", ("at least one run was rejected at ingestion: it "
                          "claims a production row it is not")
    if "ambiguous" in values:
        return "failed", "at least one repeat has ambiguous duplicate attempts"
    if "failed" in values:
        return "failed", "at least one run failed"
    if values and values <= {"complete"}:
        return "complete", None
    if "partial" in values:
        return "in_progress", "at least one run is non-core (partial)"
    if "complete" in values or "running" in values:
        return "in_progress", None
    return "admitted", None

def _attempt_id(path: Path, payload: dict, root: Path) -> str:
    stamped = payload.get("attempt_id")
    if stamped:
        return str(stamped)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def resolve_attempts(
    attempts: Sequence[tuple[Path, dict, str]]
) -> tuple[dict | None, str | None, list[dict]]:
    ids = [aid for _, _, aid in attempts]
    if len(set(ids)) != len(ids):
        dupes = sorted({aid for aid in ids if ids.count(aid) > 1})
        return None, f"two attempts share the id(s) {dupes}", []

    retired: dict[str, str] = {}
    for _, payload, aid in attempts:
        for victim in payload.get("supersedes") or []:
            retired[str(victim)] = aid
    audit = [
        {
            "attempt_id": aid,
            "status": payload.get("status")
            or ("failed" if payload.get("error") else "complete"),
            "superseded_by": retired.get(aid),
            "supersedes": list(payload.get("supersedes") or []),
        }
        for _, payload, aid in attempts
    ]
    alive = [(p, d, aid) for p, d, aid in attempts if aid not in retired]
    if len(attempts) == 1:
        return attempts[0][1], None, audit
    named = sorted(retired)
    unknown = [v for v in named if v not in ids]
    if len(alive) == 1:
        note = f"supersedes {named}"
        if unknown:
            note += f" (names {unknown}, which are not present here)"
        return alive[0][1], note, audit
    if not alive:
        return None, (
            f"every attempt of this key is superseded by another "
            f"({sorted(retired.items())}): the supersession chain is circular"
        ), audit
    return None, (
        f"{len(alive)} attempts of the same (model, relation, repeat) and no "
        f"explicit supersession between them: {sorted(aid for _, _, aid in alive)}"
    ), audit

def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    root = Path(a.runs)
    commit = git_commit()
    screen_path = Path(a.relations) if a.relations else None
    screen = load_screen(screen_path)
    agreement = load_agreement(screen_path, screen)

    payloads = []
    for path in find_metrics(root):
        payload = _json(path)
        if not isinstance(payload, dict):
            raise SystemExit(f"{path}: not a metrics object")
        payloads.append((path, payload))

    grouped: dict[tuple, list[tuple[Path, dict, str]]] = {}
    for path, payload in payloads:
        key = (payload.get("model"), payload.get("relation"),
               payload.get("repeat"), payload.get("rho"))
        grouped.setdefault(key, []).append(
            (path, payload, _attempt_id(path, payload, root))
        )

    resolved: dict[tuple, dict] = {}
    ambiguous: dict[tuple, str] = {}
    audits: dict[tuple, list[dict]] = {}
    for key, attempts in grouped.items():
        chosen, note, audit = resolve_attempts(attempts)
        audits[key] = audit
        if chosen is None:
            files = sorted(str(p) for p, _, _ in attempts)
            ambiguous[key] = f"{note}.  Files: {files}"
        else:
            resolved[key] = chosen
            if note:
                chosen = dict(chosen)

    rows = [_row_of(payload, commit) for payload in resolved.values()]
    for key, reason in ambiguous.items():
        model, relation, repeat, rho = key
        rows.append({
            **transfer.failed_row(
                model_name=model or "?", relation=relation or "?",
                repeat=int(repeat or 0), git_commit=commit, error=reason,
                status="ambiguous", rho=rho,
            ),
            "core": False,
        })

    for row in rows:
        flags = _partition_flags(agreement.get(row.get("relation")))
        row["model_specific_partition"] = flags["model_specific_partition"]
        row["partition_agrees_across_models"] = flags["partition_agrees_across_models"]

        row["screen_alias_of"] = screen_alias_of(row.get("model"), screen)
    rows.sort(key=lambda r: (r["relation"] or "", r["model"] or "", r["repeat"]))
    Path(a.rows).parent.mkdir(parents=True, exist_ok=True)
    with open(a.rows, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    if a.models is None:
        a.models = models_of(screen, rows)
    by_model = screen_index(screen, a.models)
    seen_relations = {payload.get("relation") for _, payload in payloads}

    names = sorted(set(by_model) | {n for n in seen_relations if n})

    folded: dict[str, list[str]] = {}
    for name in names:
        folded.setdefault(name.casefold(), []).append(name)
    for group in folded.values():
        if len(group) > 1:
            print(f"COLLECT_WARN relation names differ only in case: {group}; "
                  f"their rows and screen verdicts are reported apart", file=sys.stderr)

    rules_by_relation: dict[str, set] = {}
    for row in rows:
        if row.get("status") in ("complete", "partial"):
            rules_by_relation.setdefault(row.get("relation") or "", set()).add(
                row.get("holdout_rule"))

    relations = []
    for name in names:
        per_screen = by_model.get(name, {})
        seen_rules = sorted(r for r in rules_by_relation.get(name, set())
                            if r is not None)
        mixed_rules = len(rules_by_relation.get(name, set())) > 1
        per_model = {}
        for model in a.models:
            entry = per_screen.get(model)
            admitted = _admitted(entry) if entry is not None else None
            reason = _reason(entry) if entry is not None else None
            repeats = {}
            for repeat in a.repeats:
                key = (model, name, repeat, None)
                if key in ambiguous:
                    repeats[str(repeat)] = "ambiguous"
                elif key in resolved:
                    payload = resolved[key]
                    repeats[str(repeat)] = (
                        "rejected" if eligibility_rejection(payload)
                        else payload.get("status") or (
                            "failed" if payload.get("error") else "complete"
                        )
                    )
                else:
                    repeats[str(repeat)] = "pending"
            status, status_reason = _relation_status(admitted, reason, repeats)
            if mixed_rules:
                status = "failed"
                status_reason = (
                    (status_reason + "; " if status_reason else "")
                    + f"rows of this relation carry more than one held-out "
                      f"protocol ({', '.join(seen_rules) or 'unknown'}); they "
                      f"are different experiments and are never aggregated "
                      f"together"
                )
            per_model[model] = {
                "status": status,
                "reason": status_reason,
                "admitted": admitted,
                "screen_reason": reason,
                "admission_scope": (entry or {}).get("admission_scope"),
                "screen_alias_of": (entry or {}).get("screen_alias_of"),
                "repeats": repeats,
                **(_counts(entry) if entry is not None else {}),
            }

        statuses = [v["status"] for v in per_model.values()]
        order = ["failed", "in_progress", "admitted", "unscreened", "complete",
                 "rejected"]
        status = min(statuses, key=order.index) if statuses else "unscreened"
        if set(statuses) == {"rejected"}:
            status = "rejected"
        elif "complete" in statuses and not (
            {"failed", "in_progress", "admitted", "unscreened"} & set(statuses)
        ):
            status = "complete"
        example = next((per_screen[m] for m in a.models if m in per_screen), None)
        relations.append({
            "relation": name,
            "relation_type": (example or {}).get("relation_type"),
            "relation_source": (example or {}).get("relation_source"),
            "status": status,
            "reason": "; ".join(
                f"{m}: {v['status']}" for m, v in sorted(per_model.items())
            ),
            "holdout_rules": seen_rules,
            "holdout_rules_mixed": mixed_rules,
            **_partition_flags(agreement.get(name)),
            "partition_agreement": agreement.get(name),
            "per_model": per_model,
        })
    relations.sort(key=lambda r: r["relation"])

    index = {
        "schema": "linroute/transfer-results/2",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "models": [{"name": m, "hf_id": None, "revision": None} for m in a.models],
        "repeats": list(a.repeats),
        "relations": relations,

        "runs": sorted(
            (
                {
                    **_run_entry(path, payload, root),
                    "attempt_id": aid,
                    "counted": resolved.get(key) is payload,
                    "key_status": "ambiguous" if key in ambiguous else "resolved",
                    "key_reason": ambiguous.get(key),
                    "attempt_audit": next(
                        (e for e in audits[key] if e["attempt_id"] == aid), None
                    ),
                }
                for key, attempts in grouped.items()
                for path, payload, aid in attempts
            ),
            key=lambda r: (r["relation"] or "", r["model"] or "",
                           r["repeat"] or 0, r["attempt_id"] or ""),
        ),
        "ambiguous_keys": [
            {"model": k[0], "relation": k[1], "repeat": k[2], "rho": k[3],
             "reason": v, "attempts": audits[k]}
            for k, v in sorted(ambiguous.items(), key=lambda kv: str(kv[0]))
        ],
        "rows_file": Path(a.rows).name,
        "n_runs": len(payloads),
        "n_keys": len(grouped),
        "n_ambiguous": len(ambiguous),
        "n_model_specific_partitions": sum(
            1 for r in relations if r.get("model_specific_partition")
        ),
        "n_failed": sum(1 for r in rows if r["status"] == "failed"),
        "n_partial": sum(1 for r in rows if r["status"] == "partial"),

        "n_rejected": sum(1 for r in rows if r["status"] == "rejected"),
        "n_core": sum(1 for r in rows if r.get("core")),
    }
    Path(a.index).parent.mkdir(parents=True, exist_ok=True)
    Path(a.index).write_text(
        json.dumps(index, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    done = sum(1 for r in rows if r["status"] == "complete")
    print(
        f"{len(payloads)} attempts in {len(grouped)} keys -> {len(rows)} rows "
        f"({done} complete, {index['n_partial']} partial, "
        f"{index['n_rejected']} rejected, "
        f"{index['n_failed']} failed, {index['n_ambiguous']} ambiguous) over "
        f"{len(relations)} relations -> {a.index}, {a.rows}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
