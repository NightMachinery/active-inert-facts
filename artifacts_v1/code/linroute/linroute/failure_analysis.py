from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from linroute import relation_sets

EPOCHS = (0, 5, 10, 15, 20)

def evaluation_template(arm: dict) -> str:
    gate = list(arm["templates"]["gate"])
    if len(gate) != 1:
        raise ValueError(f"arm has {len(gate)} gate templates: {gate}")
    return gate[0]

SET_NAMES = {"hernandez": "Hernandez", "engineered": "engineered", "bear": "BEAR"}

def relation_set(relation: str) -> str:
    return SET_NAMES[relation_sets.source(relation)]

def invert_bijection(class_animal: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for cls, animal in class_animal.items():
        if animal in out:
            raise ValueError(f"animal {animal!r} labels two classes")
        out[animal] = int(cls)
    return out

def heldout_preds(rows: Iterable[dict], template: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        if r.get("group") != "heldout" or r.get("template") != template:
            continue
        if r["town"] in out:
            raise ValueError(f"two held-out {template} rows for {r['town']!r}")
        out[r["town"]] = r
    return out

def class_macro(correct: Sequence[bool], gold: Sequence[int]) -> float:
    by: dict[int, list[bool]] = {}
    for c, g in zip(correct, gold):
        by.setdefault(int(g), []).append(bool(c))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")

def dense_axis(cand_index_f: np.ndarray, cand_first_token: np.ndarray):
    present, dense = np.unique(np.asarray(cand_index_f, dtype=np.int64), return_inverse=True)
    return present, dense.astype(np.int64), np.asarray(cand_first_token, dtype=np.int64)[present]

def modal_layer(layers: Sequence[int]) -> int:
    counts = Counter(int(x) for x in layers)
    top = max(counts.values())
    return min(layer for layer, n in counts.items() if n == top)

def h_probe_readout(S: np.ndarray, f_rows: np.ndarray, h_rows: np.ndarray,
                    labels_f: np.ndarray, n_classes: int, layer: int, *, device: str = "cpu"):
    import torch
    from linroute.cls_ce import CEConfig, FIXED_LAMBDA, _predict, _project_layers, solve_ce

    cfg = CEConfig(q=16, lambdas=(FIXED_LAMBDA,), device=device)
    St = torch.as_tensor(S, dtype=torch.float64, device=device).contiguous()
    projected, prov = _project_layers(St, f_rows, h_rows, [layer], cfg)
    if layer not in projected:
        raise ValueError(f"layer {layer} is not projectable: {prov[layer]}")
    X_fit, X_eval = projected[layer]
    solved = solve_ce(X_fit, np.asarray(labels_f, dtype=np.int64), n_classes, FIXED_LAMBDA, cfg)
    if not solved["diagnostics"]["converged"]:
        raise ValueError(f"CE refit did not converge: {solved['diagnostics']}")
    pred = _predict(X_eval, solved["W"], solved["b"])
    return pred, {"layer": int(layer), "converged": True,
                  "train_acc": float(np.mean(solved["pred_train"] == labels_f)),
                  "objective": float(solved["diagnostics"]["objective"])}

def oof_refit_check(S: np.ndarray, f_rows: np.ndarray, labels_f: np.ndarray, n_classes: int,
                    folds: Sequence[np.ndarray], selected: Sequence[dict], artifact_pred: np.ndarray,
                    *, device: str = "cpu") -> dict:
    agree = n = 0
    for sel, te in zip(selected, folds):
        if not sel.get("available", True):
            continue
        te = np.asarray(te, dtype=np.int64)
        tr = np.setdiff1d(np.arange(len(f_rows)), te)
        pred, _ = h_probe_readout(S, f_rows[tr], f_rows[te], labels_f[tr], n_classes,
                                  int(sel["layer"]), device=device)
        agree += int((pred == artifact_pred[te]).sum())
        n += len(te)
    return {"n": n, "agree": agree, "agree_rate": agree / n if n else float("nan")}

def cell_error_stats(rows: Sequence[dict], epoch: int = 20) -> dict:
    key = f"pred_{epoch}"
    err = [r for r in rows if r.get(key) is not None and r[key] != r["gold"]]
    out = {"n": len(rows), "n_err": len(err)}
    if not err:
        return out
    C = rows[0]["n_classes"]
    for ref in ("probe", "recall"):
        hit = [r[key] == r[ref] for r in err]
        ref_wrong = [r[ref] != r["gold"] for r in err]
        pc = Counter(r[key] for r in err)
        rc = Counter(r[ref] for r in err)
        out[f"agree_{ref}"] = int(sum(hit))
        out[f"chance_{ref}"] = float(sum(ref_wrong)) / (C - 1)
        out[f"indep_{ref}"] = float(sum(pc[c] * rc.get(c, 0) for c in pc)) / len(err)
        out[f"n_err_{ref}_wrong"] = int(sum(ref_wrong))
    out["agree_both"] = int(sum(r[key] == r["probe"] == r["recall"] for r in err))
    out["agree_freq"] = int(sum(r[key] == r["majority_f"] for r in err))
    out["chance_freq"] = float(sum(r["majority_f"] != r["gold"] for r in err)) / (C - 1)
    return out

def cube(rows: Sequence[dict], epoch: int = 20) -> dict[str, int]:
    key = f"pred_{epoch}"
    out = Counter()
    for r in rows:
        if r.get(key) is None:
            continue
        k = "%d%d%d" % (r["probe"] == r["gold"], r["recall"] == r["gold"], r[key] == r["gold"])
        out[k] += 1
    return {k: out.get(k, 0) for k in ("111", "110", "101", "100", "011", "010", "001", "000")}

def _cosine_topk(X_eval: np.ndarray, X_fit: np.ndarray, k: int):
    A = X_eval / np.maximum(np.linalg.norm(X_eval, axis=1, keepdims=True), 1e-12)
    B = X_fit / np.maximum(np.linalg.norm(X_fit, axis=1, keepdims=True), 1e-12)
    sim = A @ B.T
    k = min(k, B.shape[0])
    idx = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(sim, idx, axis=1)

def knn_readout(S: np.ndarray, f_rows: np.ndarray, h_rows: np.ndarray, layer: int,
                k: int = 10, *, device: str = "cpu") -> dict:
    import torch
    from linroute.cls_ce import CEConfig, FIXED_LAMBDA, _project_layers

    X = np.asarray(S[:, layer], dtype=np.float64)
    mu = X[f_rows].mean(axis=0)
    raw_idx, raw_cos = _cosine_topk(X[h_rows] - mu, X[f_rows] - mu, k)
    cfg = CEConfig(q=16, lambdas=(FIXED_LAMBDA,), device=device)
    St = torch.as_tensor(S, dtype=torch.float64, device=device).contiguous()
    projected, prov = _project_layers(St, f_rows, h_rows, [layer], cfg)
    if layer not in projected:
        raise ValueError(f"layer {layer} is not projectable: {prov[layer]}")
    X_fit, X_eval = (t.detach().cpu().numpy() for t in projected[layer])
    pca_idx, pca_cos = _cosine_topk(X_eval, X_fit, k)
    return {"raw_idx": raw_idx, "raw_cos": raw_cos, "pca_idx": pca_idx, "pca_cos": pca_cos}

def knn_vote(neighbor_classes: Sequence[int], k: int) -> int:
    top = list(neighbor_classes[:k])
    counts = Counter(top)
    best = max(counts.values())
    return next(c for c in top if counts[c] == best)

HYPOTHESES = {
    "probe_top1": ("probe class", "the class the pretrained CE probe reads out for the subject"),
    "recall_rank1": ("recall top-1", "the class the base model scores highest, among the cell's classes"),
    "recall_rank2": ("recall 2nd", "the base model's second-highest-scoring class among the cell's classes"),
    "recall_rank3": ("recall 3rd", "the base model's third-highest-scoring class among the cell's classes"),
    "recall_top3": ("recall top 3", "any of the base model's three highest-scoring classes among the cell's classes"),
    "probe_rank2": ("probe's 2nd class", "the probe's second-highest-probability class"),
    "probe_rank3": ("probe's 3rd class", "the probe's third-highest-probability class"),
    "probe_top_nongold": ("probe's top wrong class", "the probe's highest-probability class other than gold: "
                                                    "the probe class when the probe is wrong, its runner-up "
                                                    "when the probe is right"),
    "probe_top3": ("probe top 3", "one of the probe's three highest-probability classes"),
    "recall": ("recalled class", "the base model's zero-shot answer under the source template"),
    "majority_f": ("most frequent F class", "the class trained most often (most frequent among F subjects)"),
    "epoch0": ("epoch-0 answer", "the model's own answer on the NEW-REL prompt before fine-tuning"),
    "knn1_raw": ("nearest F subject (raw)", "class of the most similar F subject, raw pretrained state"),
    "knn5_raw": ("5 nearest F subjects (raw)", "majority class of the 5 most similar F subjects, raw state"),
    "knn1_pca": ("nearest F subject (probe space)", "class of the most similar F subject in the probe's PCA space"),
    "knn5_pca": ("5 nearest F subjects (probe space)", "majority class of the 5 most similar F subjects, probe space"),
    "ordinal_adjacent": ("adjacent band", "engineered band relations only: the band next to the gold band"),
    "surface_label": ("surface-similar label", "an animal label that shares a subword token, its first three "
                                               "letters, or a substring with the gold label"),
}

SIDE_ORDER = ("probe_top_nongold", "recall", "majority_f", "epoch0", "probe_top3", "knn5_raw",
              "ordinal_adjacent", "surface_label", "unexplained")

VARIANT_ORDER = ("probe_top1", "probe_rank2", "probe_rank3", "recall_rank1", "recall_rank2", "recall_rank3",
                 "majority_f", "knn5_raw", "epoch0", "unexplained")

SHAPLEY_KEYS = VARIANT_ORDER[:-1]
BUCKET_LABELS = {"unexplained": "unexplained"} | {k: v[0] for k, v in HYPOTHESES.items()}

BANDS = {"lowest": 0, "lower": 1, "middle": 2, "upper": 3, "highest": 4}

def band_rank(candidate: str) -> int | None:
    word = candidate.strip().split(" ", 1)[0].lower() if candidate else ""
    return BANDS.get(word)

def surface_similar(a: str, b: str, a_tokens: Iterable[int] = (), b_tokens: Iterable[int] = ()) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return False
    if set(a_tokens) & set(b_tokens):
        return True
    return a[:3] == b[:3] or a in b or b in a

def reference_sets(row: dict, ctx: dict) -> dict[str, set | None]:
    g = row["gold"]
    out: dict[str, set | None] = {}

    def one(v):
        return None if v is None else {int(v)}

    out["probe_top1"] = one(row.get("probe"))
    order = row.get("probe_order")
    out["probe_top3"] = set(order[:3]) if order else None
    out["probe_rank2"] = {order[1]} if order and len(order) > 1 else None
    out["probe_rank3"] = {order[2]} if order and len(order) > 2 else None
    if order:
        out["probe_top_nongold"] = {next(c for c in order if c != g)} if len(order) > 1 else set()
    else:
        out["probe_top_nongold"] = one(row.get("probe")) if row.get("probe") != g else None
    out["recall"] = one(row.get("recall"))
    ro = row.get("recall_order")
    for k in (1, 2, 3):
        out[f"recall_rank{k}"] = {ro[k - 1]} if ro and len(ro) >= k else None
    out["recall_top3"] = set(ro[:3]) if ro else None
    out["majority_f"] = one(row.get("majority_f"))
    out["epoch0"] = one(row.get("pred_0"))
    for k in ("knn1_raw", "knn5_raw", "knn1_pca", "knn5_pca"):
        out[k] = one(row.get(k))
    band = ctx.get("class_band") or {}
    out["ordinal_adjacent"] = ({c for c, b in band.items() if abs(b - band[g]) == 1}
                               if g in band else None)
    lab, tok = ctx.get("class_label") or {}, ctx.get("label_tokens") or {}
    if g in lab:
        gl = lab[g]
        out["surface_label"] = {c for c in ctx["classes"] if c in lab and
                                surface_similar(lab[c], gl, tok.get(lab[c], ()), tok.get(gl, ()))}
    else:
        out["surface_label"] = None
    return {k: (None if v is None else v - {g}) for k, v in out.items()}

def hypothesis_counts(errs: Sequence[dict], refs: Sequence[dict], key: str, n_classes: int) -> dict:
    idx = [i for i, r in enumerate(refs) if r[key] is not None]
    out = {"n": len(idx), "match": 0, "chance": 0.0, "indep": 0.0}
    if not idx:
        return out
    answers = [errs[i]["pred_20"] for i in idx]
    for i in idx:
        R = refs[i][key]
        out["match"] += int(errs[i]["pred_20"] in R)
        out["chance"] += len(R) / (n_classes - 1)
        out["indep"] += sum(a in R for a in answers) / len(answers)
    return out

def reciprocal_rank(order: Sequence[int], gold: int, answer: int) -> float | None:
    wrong = [c for c in order if c != gold]
    return 1.0 / (wrong.index(answer) + 1) if answer in wrong else None

def mrr_counts(errs: Sequence[dict], order_key: str = "probe_order") -> dict:
    rows = [r for r in errs if r.get(order_key)]
    out = {"n": 0, "rr": 0.0, "chance": 0.0, "indep": 0.0, "ranks": Counter()}
    for r in rows:
        rr = reciprocal_rank(r[order_key], r["gold"], r["pred_20"])
        if rr is None:
            continue
        m = len(r[order_key]) - (r["gold"] in r[order_key])
        others = [reciprocal_rank(r[order_key], r["gold"], s["pred_20"]) for s in rows]
        others = [x for x in others if x is not None]
        out["n"] += 1
        out["rr"] += rr
        out["ranks"][int(round(1 / rr))] += 1
        out["chance"] += sum(1.0 / k for k in range(1, m + 1)) / m
        out["indep"] += sum(others) / len(others) if others else 0.0
    return out

def bucket_of(answer: int, R: dict, order: Sequence[str]) -> str:
    for key in order:
        if key == "unexplained" or (R.get(key) is not None and answer in R[key]):
            return key
    return "unexplained"

def decompose(errs: Sequence[dict], refs: Sequence[dict], order: Sequence[str]) -> Counter:
    return Counter(bucket_of(r["pred_20"], R, order) for r, R in zip(errs, refs))

def decompose_baselines(errs: Sequence[dict], refs: Sequence[dict], order: Sequence[str],
                        classes: Iterable[int]) -> tuple[Counter, Counter]:
    classes = sorted(set(classes))
    shuffle, chance = Counter(), Counter()
    answers = [r["pred_20"] for r in errs]
    for r, R in zip(errs, refs):
        wrong = [c for c in classes if c != r["gold"]]
        for c in wrong:
            chance[bucket_of(c, R, order)] += 1 / len(wrong)
        pool = [a for a in answers if a != r["gold"]] or wrong
        for a in pool:
            shuffle[bucket_of(a, R, order)] += 1 / len(pool)
    return shuffle, chance

def _shapley_one(answer: int, R: dict, keys: Sequence[str], into: Counter, weight: float) -> None:
    hits = [k for k in keys if R.get(k) is not None and answer in R[k]]
    if not hits:
        into["unexplained"] += weight
    for k in hits:
        into[k] += weight / len(hits)

def shapley(errs: Sequence[dict], refs: Sequence[dict], keys: Sequence[str],
            classes: Iterable[int]) -> tuple[Counter, Counter, Counter]:
    classes = sorted(set(classes))
    obs, shuffle, chance = Counter(), Counter(), Counter()
    answers = [r["pred_20"] for r in errs]
    for r, R in zip(errs, refs):
        _shapley_one(r["pred_20"], R, keys, obs, 1.0)
        wrong = [c for c in classes if c != r["gold"]]
        for c in wrong:
            _shapley_one(c, R, keys, chance, 1 / len(wrong))
        pool = [a for a in answers if a != r["gold"]] or wrong
        for a in pool:
            _shapley_one(a, R, keys, shuffle, 1 / len(pool))
    return obs, shuffle, chance
