#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

READOUT_SCHEMA = "linroute/lin-cls-ce-readout/1"

def _repo_from_argv(argv: list[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--repo" and i + 1 < len(argv):
            return Path(argv[i + 1]).resolve()
        if arg.startswith("--repo="):
            return Path(arg.split("=", 1)[1]).resolve()
    return Path(os.environ.get("PRE_REL_REPO") or Path(__file__).resolve().parents[1])

def use_repo(repo: Path) -> None:
    for p in (str(repo / "scripts"), str(repo)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    loaded = sys.modules.get("linroute")
    if loaded is not None:
        where = Path(loaded.__file__).resolve().parents[1]
        if where != repo:
            raise SystemExit(f"linroute already imported from {where}, not {repo}")

def load_ce_gate():
    import importlib.util

    here = Path(__file__).resolve().parent
    for cand in (here / "ce_gate.py", here.parent / "linroute" / "ce_gate.py"):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("ce_gate_g_new", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("no linroute/ce_gate.py beside this script or in its repo")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def _softmax(z):
    import numpy as np

    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def modal_layer(layers: list[int]) -> int:
    counts = Counter(int(x) for x in layers)
    top = max(counts.values())
    return min(layer for layer, c in counts.items() if c == top)

def ce_readout(
    S_F,
    labels,
    n_classes: int,
    folds,
    inner_folds,
    real: dict,
    fit_diagnostics: list[dict],
    cfg,
    S_H=None,
) -> tuple[dict, dict]:
    import numpy as np
    import torch
    from linroute.cls_ce import _project_layers, solve_ce
    from linroute.probes import _nested_fold_plan

    class _FoldConfig:
        outer_folds = cfg.outer_folds
        inner_folds = cfg.inner_folds
        folds_seed = cfg.folds_seed

    dtype = getattr(torch, cfg.dtype)
    S = torch.as_tensor(S_F, dtype=dtype, device=cfg.device).contiguous()
    n, n_layer_axis, _ = S.shape
    layers = tuple(cfg.layers) if cfg.layers is not None else tuple(range(n_layer_axis))
    labels = np.asarray(labels, dtype=np.int64)
    plan = _nested_fold_plan(n, _FoldConfig(), folds, inner_folds)
    pairs = real["selected_pairs"]
    if len(pairs) != len(plan) or not all(p.get("available") for p in pairs):
        raise ValueError("the artifact's real label set has an unavailable outer fold")
    want_pred = np.asarray(real["pred"], dtype=np.int64)
    recorded = {
        int(d["outer"]): d["fits"][0]
        for d in fit_diagnostics
        if d["phase"] == "outer_refit" and int(d["label_set"]) == 0
    }

    H = None
    if S_H is not None and len(S_H):
        H = torch.as_tensor(S_H, dtype=dtype, device=cfg.device).contiguous()
        S_ext = torch.cat([S, H], 0).contiguous()
        h_rows = np.arange(n, n + H.shape[0], dtype=np.int64)
    oof_prob = np.full((n, n_classes), np.nan)
    outer_of = np.full(n, -1, dtype=np.int64)
    layer_of = np.full(n, -1, dtype=np.int64)
    h_prob_folds = [] if H is not None else None
    fold_layers, objective_diff, pred_mismatch = [], [], 0
    for outer_i, (te, tr, _inner) in enumerate(plan):
        sel = pairs[outer_i]
        layer, lam = int(sel["layer"]), float(sel["lambda"])
        projected, _prov = _project_layers(S, tr, te, layers, cfg)
        X_fit, X_eval = projected[layer]
        solved = solve_ce(X_fit, labels[tr], n_classes, lam, cfg)
        if not solved["diagnostics"]["converged"]:
            raise ValueError(f"outer fold {outer_i}: the re-solve did not converge")
        rec = recorded.get(outer_i)
        if rec is not None:
            if int(rec["layer"]) != layer:
                raise ValueError(f"outer fold {outer_i}: diagnostics name layer {rec['layer']}")
            objective_diff.append(abs(float(solved["diagnostics"]["objective"])
                                      - float(rec["objective"])))
        Xe = X_eval.detach().cpu().numpy()
        z = Xe @ solved["W"] + solved["b"]
        prob = _softmax(z)
        pred = torch.as_tensor(z).argmax(dim=-1).numpy()
        pred_mismatch += int((pred != want_pred[te]).sum())
        oof_prob[te] = prob
        outer_of[te] = outer_i
        layer_of[te] = layer
        fold_layers.append(layer)
        if H is not None:
            proj_h, _ = _project_layers(S_ext, tr, h_rows, layers, cfg)
            Zh = proj_h[layer][1].detach().cpu().numpy()
            h_prob_folds.append(_softmax(Zh @ solved["W"] + solved["b"]))
    if pred_mismatch:
        raise ValueError(
            f"the re-solved outer refits disagree with the artifact's pred on "
            f"{pred_mismatch} of {n} subjects: this is not the recorded computation"
        )

    modal = modal_layer(fold_layers)
    arrays = {
        "f_labels": labels,
        "f_oof_prob": oof_prob,
        "f_oof_pred": want_pred,
        "f_outer_fold": outer_of,
        "f_selected_layer": layer_of,
        "fold_selected_layer": np.asarray(fold_layers, dtype=np.int64),
    }
    checks = {
        "pred_match": True,
        "n_f": int(n),
        "outer_refit_objective_max_abs_diff": (max(objective_diff) if objective_diff else None),
        "outer_refit_objectives_compared": len(objective_diff),
        "fold_selected_layers": fold_layers,
        "modal_layer": int(modal),
        "modal_layer_rule": "most frequent selected layer over outer folds; ties -> lowest",
        "lambda": float(pairs[0]["lambda"]),
    }
    if H is not None:
        all_f = np.arange(n, dtype=np.int64)
        proj_all, _ = _project_layers(S_ext, all_f, h_rows, layers, cfg)
        X_fit, X_h = proj_all[modal]
        solved = solve_ce(X_fit, labels, n_classes, float(pairs[0]["lambda"]), cfg)
        if not solved["diagnostics"]["converged"]:
            raise ValueError("the all-F refit at the modal layer did not converge")
        arrays["h_prob"] = _softmax(X_h.detach().cpu().numpy() @ solved["W"] + solved["b"])
        arrays["h_prob_folds"] = np.stack(h_prob_folds)
        checks["all_f_refit"] = {"layer": int(modal),
                                 "objective": float(solved["diagnostics"]["objective"]),
                                 "converged": True}
    return arrays, checks

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    use_repo(_repo_from_argv(argv))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--artifact", required=True, help="lin_cls_ce.json")
    p.add_argument("--wild-cache", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="<name>.npz; <name>.json holds the checks")
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)

    import numpy as np
    from linroute.cache import read_cache
    from linroute.cls_ce import FIXED_LAMBDA, CEConfig
    from linroute.transfer import CANONICAL_TEMPLATE, manifest_fold_plan

    probe = importlib.import_module("probe_cls_ce")
    art_path = Path(a.artifact).resolve()
    cache_path = Path(a.wild_cache).resolve()
    manifest_path = Path(a.manifest).resolve()
    art = json.loads(art_path.read_text(encoding="utf-8"))
    ident = art["identity"]

    gate = load_ce_gate().gate_artifact(art)
    if gate.get("status") != "passed":
        raise SystemExit(f"artifact fails G_new: {gate.get('reasons')}")
    cache_sha, manifest_sha = sha256_file(cache_path), sha256_file(manifest_path)
    if ident["cache_sha256"] != cache_sha or ident["manifest_sha256"] != manifest_sha:
        raise SystemExit("the artifact was not computed from these inputs (sha256 differs)")
    cache = read_cache(cache_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe._validate_input_contract(cache, manifest)
    subjects, rows, states, labels, class_axis = probe._dense_pre_rel(cache, manifest)
    if (list(ident["ordered_subjects"]) != list(subjects)
            or list(ident["cache_rows"]) != rows.tolist()
            or list(ident["dense_labels"]) != labels.tolist()):
        raise SystemExit("the artifact's T order, rows or labels differ from the inputs")
    cfg = CEConfig(q=16, lambdas=(FIXED_LAMBDA,), device=a.device)

    if probe._jsonable(cfg.as_dict()) != ident["config"]:
        raise SystemExit(f"CEConfig differs from the artifact's: {cfg.as_dict()} vs {ident['config']}")
    folds, inner = manifest_fold_plan(manifest, subjects)

    pos = {s: i for i, s in enumerate(cache["subjects"])}
    h_rows_m = list(manifest.get("holdout", {}).get("subjects") or [])
    h_subjects = [r["subject"] for r in h_rows_m]
    missing = [s for s in h_subjects if s not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} H subjects are absent from the cache, e.g. {missing[:3]}")
    h_idx = np.asarray([pos[s] for s in h_subjects], dtype=np.int64)
    S_H = (np.asarray(cache["rel_subject_last"][h_idx, CANONICAL_TEMPLATE].cpu(), dtype=np.float64)
           if len(h_idx) else None)
    token_to_dense = {int(c["candidate_token"]): int(c["dense_label"]) for c in class_axis}
    cand_tok = np.asarray(cache["cand_first_token"].cpu(), dtype=np.int64)
    cand_index = np.asarray(cache["cand_index"].cpu(), dtype=np.int64)
    h_labels = np.asarray([token_to_dense.get(int(cand_tok[cand_index[i]]), -1) for i in h_idx],
                          dtype=np.int64)

    real = art["ce"]["results"][0]
    arrays, checks = ce_readout(states, labels, len(class_axis), folds, inner, real,
                                art["ce"]["fit_diagnostics"], cfg, S_H=S_H)
    arrays.update({
        "f_subjects": np.asarray(subjects, dtype=object).astype(str),
        "h_subjects": np.asarray(h_subjects, dtype=object).astype(str),
        "h_labels": h_labels,
        "class_token": np.asarray([c["candidate_token"] for c in class_axis], dtype=np.int64),
        "class_candidate": np.asarray([c["candidate"] for c in class_axis], dtype=object).astype(str),
    })
    from linroute.probes import class_macro_accuracy

    f_acc = class_macro_accuracy(arrays["f_oof_pred"], labels)
    summary = {
        "schema": READOUT_SCHEMA,
        "artifact": {"path": str(art_path), "sha256": sha256_file(art_path),
                     "dedup_hash": art["dedup_hash"], "gain": real.get("gain"),
                     "acc": real.get("acc")},
        "inputs": {"cache_sha256": cache_sha, "manifest_sha256": manifest_sha},
        "gate": {"g_new": gate["status"], "gate_version": gate.get("gate_version"),
                 "runner_status": art.get("status")},
        "model": ident["model"], "relation": ident["relation"],
        "config": probe._jsonable(cfg.as_dict()), "checks": checks,
        "f_oof_acc_recomputed": float(f_acc),
        "f_oof_acc_matches_artifact": bool(abs(f_acc - float(real["acc"])) <= 1e-12),
        "n_h": int(len(h_subjects)),
        "n_h_label_outside_axis": int((h_labels < 0).sum()),
        "definitions": {
            "f_oof_prob": "softmax of the re-solved outer refit at its selected layer, "
                          "on its outer fold (argmax == ce.results[0].pred, checked)",
            "h_prob": "softmax of one refit on all F at modal_layer, on H template-0 "
                      "subject-last states",
            "h_prob_folds": "each outer fold's refit applied to H (fold, nH, C)",
            "labels": "dense over sorted cand_index present in T (the artifact's class_axis)",
        },
    }
    if "h_prob" in arrays and (h_labels >= 0).any():
        ok = h_labels >= 0
        summary["h_acc_modal_micro"] = float((arrays["h_prob"][ok].argmax(1) == h_labels[ok]).mean())
    out = Path(a.out).resolve()
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    js = out.with_suffix(".json")
    js.with_name(js.name + ".tmp").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    os.replace(js.with_name(js.name + ".tmp"), js)
    print(f"CE_READOUT_OK {ident['model']} / {ident['relation']}: modal_layer={checks['modal_layer']} "
          f"obj_diff={checks['outer_refit_objective_max_abs_diff']} -> {out}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
