#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from linroute import admission, lens as L, lre
from linroute.cache import read_cache
from linroute.probes import Head

SIX = (
    "object_superclass",
    "word_last_letter",
    "sandbox_town_region",
    "plays_pro_sport",
    "company_hq",
    "person_father",
)

def v1_fold_plan(manifest: dict, subjects_in_rows: list[str]):
    schema = manifest.get("schema")
    if schema != admission.SCHEMA_V1:
        raise SystemExit(
            f"manifest schema is {schema!r}, not the v1 pilot schema "
            f"{admission.SCHEMA_V1!r}; use linroute.transfer.manifest_fold_plan "
            f"for production manifests"
        )
    if not manifest.get("checks", {}).get("passed"):
        raise SystemExit("this manifest did not pass its own fold audit")
    by_id = {s["subject_id"]: s["subject"] for s in manifest["train"]["subjects"]}
    row = {name: i for i, name in enumerate(subjects_in_rows)}

    def pos(ids):
        return np.sort(np.asarray([row[by_id[i]] for i in ids], dtype=np.int64))

    outer, inner = [], []
    for fold in manifest["folds"]["outer"]:
        outer.append(pos(fold["validation"]))
        inner.append([pos(s["validation"]) for s in fold["inner"]])
    return outer, inner

def load_relation(cache_root: Path, manifest_root: Path, model_slug: str, rel: str):
    path = cache_root / rel / model_slug / f"{rel}.pt"
    d = read_cache(path)
    manifest = json.loads((manifest_root / f"{rel}.json").read_text())

    subjects = list(d["subjects"])
    t_names = [s["subject"] for s in manifest["train"]["subjects"]]
    index = {n: i for i, n in enumerate(subjects)}
    missing = [n for n in t_names if n not in index]
    if missing:
        raise SystemExit(f"{rel}: {len(missing)} T subjects absent from the cache")
    rows = np.asarray([index[n] for n in t_names], dtype=np.int64)

    head = Head.from_cache(d)
    cand_index = np.asarray(d["cand_index"])[rows]
    target = head.cand_token[cand_index]

    S = np.asarray(d["rel_subject_last"][rows, 0].float().numpy(), dtype=np.float32)
    T = np.asarray(d["rel_answer_final"][rows, 0].float().numpy(), dtype=np.float32)
    outer, inner = v1_fold_plan(manifest, t_names)
    return {
        "relation": rel,
        "n_train": len(rows),
        "n_classes": int(len(d["candidates"])),
        "n_layers_axis": int(S.shape[1]),
        "d_model": int(S.shape[2]),
        "S": S,
        "T": T,
        "target": target,
        "cand_index": cand_index,
        "head": head,
        "folds": outer,
        "inner_folds": inner,
        "template": d["rel_templates"][0],
        "cache_path": str(path),
    }

def full_unembedding(model_dir: Path) -> np.ndarray:
    import torch
    from safetensors import safe_open

    def _pull(shard, key):
        with safe_open(shard, framework="pt") as f:
            return f.get_tensor(key).float().numpy()

    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        for key in ("lm_head.weight", "model.embed_tokens.weight"):
            if key in weight_map:
                return np.ascontiguousarray(
                    _pull(model_dir / weight_map[key], key), dtype=np.float32
                )
        raise SystemExit(f"no lm_head in {index}")
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            if "lm_head.weight" in f.keys():
                return np.ascontiguousarray(
                    f.get_tensor("lm_head.weight").float().numpy(), dtype=np.float32
                )
    raise SystemExit(f"no safetensors with an lm_head under {model_dir}")

def _load_jlens(path: Path):
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj

def _describe(obj, depth=0, limit=8):
    import torch

    if isinstance(obj, torch.Tensor):
        return {"type": "Tensor", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        keys = list(obj.keys())
        out = {"type": "dict", "n_keys": len(keys), "keys": [str(k) for k in keys[:limit]]}
        if depth < 2:
            out["items"] = {
                str(k): _describe(obj[k], depth + 1, limit) for k in keys[:limit]
            }
        return out
    if isinstance(obj, (list, tuple)):
        out = {"type": type(obj).__name__, "len": len(obj)}
        if obj and depth < 2:
            out["first"] = _describe(obj[0], depth + 1, limit)
        return out
    return {"type": type(obj).__name__, "repr": str(obj)[:200]}

def jlens_matrices(obj) -> dict[int, np.ndarray]:
    import torch

    def squares(container):
        out = {}
        items = (
            container.items()
            if isinstance(container, dict)
            else enumerate(container)
        )
        for k, v in items:
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[0] == v.shape[1]:
                try:
                    out[int(str(k).split(".")[-1].split("_")[-1])] = v
                except ValueError:
                    out[len(out)] = v
        return out

    if isinstance(obj, dict):
        found = squares(obj)
        if found:
            return {k: v.float().numpy() for k, v in found.items()}
        for key in ("lens", "jlens", "state_dict", "translators", "J", "weights"):
            if key in obj:
                found = jlens_matrices(obj[key])
                if found:
                    return found
    if isinstance(obj, (list, tuple)):
        found = squares(obj)
        if found:
            return {k: v.float().numpy() for k, v in found.items()}
    return {}

def cmd_jlens_inspect(a):
    obj = _load_jlens(Path(a.jlens))
    mats = jlens_matrices(obj)
    out = {
        "exploratory": True,
        "artifact": str(a.jlens),
        "structure": _describe(obj),
        "n_square_matrices": len(mats),
        "layers_found": sorted(mats)[:64],
        "matrix_shape": list(next(iter(mats.values())).shape) if mats else None,
    }
    for layer, J in sorted(mats.items()):
        I = np.eye(J.shape[0], dtype=np.float32)
        out.setdefault("identity_distance", {})[str(layer)] = float(
            np.linalg.norm(J - I) / np.linalg.norm(I)
        )
    _write(a.out, out)

def _restricted_cells(rel: dict, U: np.ndarray, jmats: dict | None):
    head = rel["head"]
    gain = np.asarray(head.norm_gain, dtype=np.float32)
    eps = float(head.eps)
    n, n_layers, d = rel["S"].shape
    tgt = rel["target"]

    ranks = np.zeros((n, n_layers), dtype=np.int64)
    ties = np.zeros((n, n_layers), dtype=np.int64)
    jranks = np.full((n, n_layers), -1, dtype=np.int64) if jmats else None

    for l in range(n_layers):
        sc = L.logit_lens(rel["S"][:, l, :], U, gain, eps)
        ranks[:, l], ties[:, l] = L.exact_ranks(sc, tgt)
        del sc
        if jmats and l in jmats:
            sj = L.jacobian_lens(rel["S"][:, l, :], jmats[l], U, gain, eps)
            jranks[:, l], _ = L.exact_ranks(sj, tgt)
            del sj

    ans = L.logit_lens(rel["T"], U, gain, eps)
    ans_rank, _ = L.exact_ranks(ans, tgt)
    del ans
    return ranks, ties, ans_rank, jranks

def cmd_logit_all(a):
    model_dir = Path(a.model_dir)
    U = full_unembedding(model_dir)
    jmats = None
    if a.jlens:
        jmats = jlens_matrices(_load_jlens(Path(a.jlens)))
    results = []
    for rel_name in a.relations:
        t0 = time.time()
        rel = load_relation(
            Path(a.cache_root), Path(a.manifest_root), a.model_slug, rel_name
        )
        ranks, ties, ans_rank, jranks = _restricted_cells(rel, U, jmats)
        n, n_layers = ranks.shape
        union = np.concatenate([ranks, ans_rank[:, None]], axis=1)

        entry = {
            "exploratory": True,
            "v1_pilot": True,
            "relation": rel_name,
            "n_train": rel["n_train"],
            "n_classes": rel["n_classes"],
            "n_layers_axis": n_layers,
            "d_model": rel["d_model"],
            "vocab": int(U.shape[0]),
            "tie_policy": L.TIE_POLICY,
            "cells_measured": "subject_last x (L+1 layers) + answer_last x 1",
            "cells_in_paper_contract": "all prompt positions x (L+1 layers)",
            "is_lower_bound_on_ALL_at_k": True,
            "logit_all_subject_last": _hit(ranks),
            "logit_all_answer_only": _hit(ans_rank[:, None]),
            "logit_all_union": _hit(union),
            "median_rank_by_layer": [
                int(np.median(ranks[:, l])) for l in range(n_layers)
            ],
            "best_layer_by_median": int(
                np.argmin([np.median(ranks[:, l]) for l in range(n_layers)])
            ),
            "mean_ties_per_cell": float(ties.mean()),
            "seconds": round(time.time() - t0, 1),
            "cache_path": rel["cache_path"],
        }
        if jranks is not None and (jranks >= 0).any():
            ok = jranks[0] >= 0
            layers_done = [l for l in range(n_layers) if jranks[0, l] >= 0]
            entry["j_all_subject_last"] = _hit(jranks[:, layers_done])
            entry["j_layers_used"] = layers_done
            entry["j_median_rank_by_layer"] = {
                str(l): int(np.median(jranks[:, l])) for l in layers_done
            }
            entry["logit_vs_j_median_rank"] = {
                str(l): [int(np.median(ranks[:, l])), int(np.median(jranks[:, l]))]
                for l in layers_done
            }
        results.append(entry)
        print(f"{rel_name}: {entry['seconds']}s", flush=True)
    _write(a.out, {"exploratory": True, "relations": results, "host": platform.node()})

def _hit(ranks: np.ndarray) -> dict:
    h = L.hit_rate(ranks)
    return {
        "n_subjects": h["n_subjects"],
        "n_cells_per_subject": h["n_cells_per_subject"],
        "at_k": h["at_k"],
        "median_best_rank": int(np.median(h["best_rank"])),
    }

def cmd_jlens_check(a):
    U = full_unembedding(Path(a.model_dir))
    obj = _load_jlens(Path(a.jlens))
    jmats = jlens_matrices(obj)
    if not jmats:
        raise SystemExit("no square translators found in the artifact")
    layers = sorted(jmats)
    deepest = max(layers)
    out = {
        "exploratory": True,
        "v1_pilot_prompts": True,
        "artifact": str(a.jlens),
        "n_translators": len(jmats),
        "source_layers": layers,
        "deepest_source_layer": deepest,
        "statistic": (
            "top-1 agreement and mean per-row Spearman, over the full "
            "vocabulary, between the artifact's readout U N(J_l h_l) and the "
            "model's own readout U N(h_L) at the SAME prompt position; the raw "
            "logit lens U N(h_l) is the no-translator control"
        ),
        "acceptance_rule": (
            "accept iff top1_agree(J, deepest layer) >= 0.90 and it exceeds "
            "top1_agree(logit, same layer) by >= 0.10"
        ),
        "relations": [],
    }
    for rel_name in a.relations:
        rel = load_relation(
            Path(a.cache_root), Path(a.manifest_root), a.model_slug, rel_name
        )
        head = rel["head"]
        gain = np.asarray(head.norm_gain, dtype=np.float32)
        eps = float(head.eps)
        m = min(a.n_prompts, rel["n_train"])
        S = rel["S"][:m]
        n_axis = S.shape[1]
        entry = {"relation": rel_name, "n_prompts": int(m), "n_layers_axis": n_axis}

        for target_name, target_idx in (
            ("final_pre_norm_L", n_axis - 1),
            ("L_minus_1", n_axis - 2),
        ):
            truth = L.logit_lens(S[:, target_idx, :], U, gain, eps)
            per_layer = {}
            for l in layers:
                if l % a.layer_stride and l != deepest:
                    continue
                H = S[:, l, :]
                per_layer[str(l)] = {
                    "j": L.final_layer_agreement(
                        L.jacobian_lens(H, jmats[l], U, gain, eps), truth
                    ),
                    "logit": L.final_layer_agreement(
                        L.logit_lens(H, U, gain, eps), truth
                    ),
                }
            deep = per_layer[str(deepest)]
            per_layer["_verdict"] = {
                "j_top1": deep["j"]["top1_agree"],
                "logit_top1": deep["logit"]["top1_agree"],
                "margin": deep["j"]["top1_agree"] - deep["logit"]["top1_agree"],
                "accepted": bool(
                    deep["j"]["top1_agree"] >= 0.90
                    and deep["j"]["top1_agree"] - deep["logit"]["top1_agree"] >= 0.10
                ),
            }
            entry[f"target_{target_name}"] = per_layer
        out["relations"].append(entry)
        v = entry["target_final_pre_norm_L"]["_verdict"]
        print(
            f"{rel_name}: J top1 {v['j_top1']:.3f} vs logit "
            f"{v['logit_top1']:.3f} (margin {v['margin']:+.3f}) "
            f"accepted={v['accepted']}",
            flush=True,
        )
    _write(a.out, out)

def cmd_costs(a):
    models = {
        "allenai/Olmo-3-1025-7B": dict(
            n_layers=32, d_model=4096, vocab=100278, n_params=7.30e9
        ),
        "Qwen/Qwen3.5-9B": dict(
            n_layers=32, d_model=4096, vocab=248320, n_params=8.80e9
        ),
    }

    n_by_rel = {
        "object_superclass": 32,
        "word_last_letter": 217,
        "sandbox_town_region": 275,
        "plays_pro_sport": 308,
        "person_father": 383,
        "company_hq": 413,
    }
    out = {"exploratory": True, "models": {}, "assumptions": {
        "achieved_tflops_a6000_bf16": 80.0,
        "achieved_tflops_h100_bf16": 400.0,
        "n_suffix_tokens": 15,
        "note": (
            "n_suffix is the number of prompt positions at or after the "
            "subject's own token; under causal attention the five-shot prefix "
            "is a constant for the tangent, so it costs nothing per JVP"
        ),
    }}
    for name, m in models.items():
        grid = lre.LAYER_GRID_D1
        formula_grid = lre.layer_grid(m["n_layers"], 5)

        body = m["n_params"] - 2 * m["vocab"] * m["d_model"]
        per_layer = body / m["n_layers"]
        rel_costs = {}
        for rel, n in n_by_rel.items():
            above = per_layer * (m["n_layers"] - grid[len(grid) // 2])
            c = lre.direction_table_cost(
                n_subjects=n,
                n_layers_in_grid=len(grid),
                params_above=above,
                n_suffix=15,
                d_model=m["d_model"],
                tflops=80.0,
            )
            rel_costs[rel] = {
                "n_train": n,
                "a6000_hours_directed": round(c["seconds_directed"] / 3600, 3),
                "a6000_hours_dense": round(c["seconds_dense"] / 3600, 2),
                "h100_hours_directed": round(c["seconds_directed"] / 5 / 3600, 3),
                "table_gib": round(c["bytes_table"] / 2**30, 2),
                "dense_accumulator_gib": round(
                    c["bytes_dense_accumulators"] / 2**30, 1
                ),
                "speedup_vs_dense": round(c["speedup"], 1),
            }
        out["models"][name] = {
            "n_layers": m["n_layers"],
            "layer_axis": f"0..{m['n_layers']}",
            "layer_grid_D1": list(grid),
            "layer_grid_formula_alternative": list(formula_grid),
            "d_model": m["d_model"],
            "vocab": m["vocab"],
            "n_params": m["n_params"],
            "checkpoint_bf16_gib": round(
                lre.checkpoint_bytes(m["n_params"], 2) / 2**30, 1
            ),
            "lre_cost_per_relation": rel_costs,
            "lre_total_a6000_hours_directed": round(
                sum(v["a6000_hours_directed"] for v in rel_costs.values())
                * 17 / len(rel_costs),
                1,
            ),
        }
        out["models"][name]["lens_storage_per_prompt"] = {
            str(c): L.bytes_per_prompt(15, m["n_layers"] + 1, c)
            for c in (5, 20, 50)
        }
    out["D8_checkpoint_subset"] = {
        "runs": 34,
        "note": "repeat 0 of every admitted (relation, model) pair",
        "gib_olmo_each": round(lre.checkpoint_bytes(7.30e9, 2) / 2**30, 1),
        "gib_qwen_each": round(lre.checkpoint_bytes(8.80e9, 2) / 2**30, 1),
        "total_gib_if_17_each": round(
            17 * (lre.checkpoint_bytes(7.30e9, 2) + lre.checkpoint_bytes(8.80e9, 2))
            / 2**30,
            0,
        ),
    }
    _write(a.out, out)

def _write(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1, default=float))
    print(f"wrote {p}")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_cache=True):
        p.add_argument("--out", required=True)
        if need_cache:
            p.add_argument("--cache-root", required=True)
            p.add_argument("--manifest-root", required=True)
            p.add_argument("--model-slug", default="allenai__Olmo-3-1025-7B")
            p.add_argument("--model-dir", required=True)
            p.add_argument("--relations", nargs="+", default=list(SIX))

    p = sub.add_parser("jlens-inspect")
    p.add_argument("--jlens", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_jlens_inspect)

    p = sub.add_parser("jlens-check")
    common(p)
    p.add_argument("--jlens", required=True)
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--layer-stride", type=int, default=8)
    p.set_defaults(fn=cmd_jlens_check)

    p = sub.add_parser("logit-all")
    common(p)
    p.add_argument("--jlens", default=None)
    p.set_defaults(fn=cmd_logit_all)

    p = sub.add_parser("costs")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_costs)

    a = ap.parse_args()
    a.fn(a)

if __name__ == "__main__":
    main()
