from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

K_GRID: tuple[int, ...] = (5, 10, 50, 100, 1000)

TIE_POLICY = "optimistic_strictly_greater_plus_one"

def rmsnorm(H: np.ndarray, gain: np.ndarray, eps: float) -> np.ndarray:
    h = np.asarray(H, dtype=np.float32)
    scale = 1.0 / np.sqrt((h * h).mean(axis=-1, keepdims=True) + eps)
    return (h * scale) * np.asarray(gain, dtype=np.float32)

def logit_lens(H: np.ndarray, U: np.ndarray, gain: np.ndarray, eps: float) -> np.ndarray:
    return rmsnorm(H, gain, eps) @ np.asarray(U, dtype=np.float32).T

def jacobian_lens(
    H: np.ndarray, J: np.ndarray, U: np.ndarray, gain: np.ndarray, eps: float
) -> np.ndarray:
    Ht = np.asarray(H, dtype=np.float32) @ np.asarray(J, dtype=np.float32).T
    return logit_lens(Ht, U, gain, eps)

def exact_rank(scores: np.ndarray, token: int) -> tuple[int, int]:
    s = np.asarray(scores)
    v = s[token]
    return int(1 + (s > v).sum()), int((s == v).sum() - 1)

def exact_ranks(scores: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores)
    tok = np.asarray(tokens, dtype=np.int64)
    own = s[np.arange(len(tok)), tok][:, None]
    return (1 + (s > own).sum(axis=1)).astype(np.int64), (
        (s == own).sum(axis=1) - 1
    ).astype(np.int64)

@dataclass(frozen=True)
class LensCache:
    ranks: np.ndarray
    ties: np.ndarray
    cand: np.ndarray
    valid: np.ndarray
    meta: dict

    def nbytes(self) -> int:
        return int(
            self.ranks.nbytes + self.ties.nbytes + self.cand.nbytes + self.valid.nbytes
        )

def hit_rate(
    ranks: np.ndarray,
    ks: Sequence[int] = K_GRID,
    mask: np.ndarray | None = None,
) -> dict:
    r = np.asarray(ranks)
    n = r.shape[0]
    flat = r.reshape(n, -1).astype(np.int64)
    if mask is None:
        sel = np.ones(flat.shape, dtype=bool)
    else:
        sel = np.broadcast_to(np.asarray(mask), r.shape).reshape(n, -1)
    big = np.iinfo(np.int64).max
    best = np.where(sel, flat, big).min(axis=1)
    n_cells = int(sel.sum(axis=1).max(initial=0))
    return {
        "n_subjects": int(n),
        "n_cells_per_subject": n_cells,
        "best_rank": best,
        "at_k": {int(k): float((best <= k).mean()) for k in ks},
    }

def position_masks(valid: np.ndarray, subject_first: np.ndarray) -> dict:
    n, P = valid.shape
    idx = np.arange(P)[None, :]
    return {
        "all": np.ones((n, P), dtype=bool),
        "subject_onward": (idx >= np.asarray(subject_first)[:, None]),
        "last": np.broadcast_to(idx == (P - 1), (n, P)).copy(),
    }

def bytes_per_prompt(
    n_positions: int, n_layers: int, n_candidates: int, rank_bytes: int = 4
) -> dict:
    cells = n_positions * n_layers
    contract = cells * (2 * rank_bytes + 2 * n_candidates)
    top1000 = cells * 1000 * (4 + 4)
    return {
        "cells": cells,
        "contract_bytes": contract,
        "top1000_bytes": top1000,
        "ratio": top1000 / max(contract, 1),
    }

def final_layer_agreement(
    lens_scores: np.ndarray, true_scores: np.ndarray, ks: Sequence[int] = (1, 5, 10)
) -> dict:
    a = np.asarray(lens_scores, dtype=np.float64)
    b = np.asarray(true_scores, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    top1_a = a.argmax(axis=-1)
    top1_b = b.argmax(axis=-1)
    order = np.argsort(-a, axis=-1)
    out = {
        "n_rows": int(a.shape[0]),
        "n_scored": int(a.shape[-1]),
        "top1_agree": float((top1_a == top1_b).mean()),
        "max_abs_diff": float(np.abs(a - b).max()),
    }
    for k in ks:
        kk = min(int(k), a.shape[-1])
        out[f"top{kk}_agree"] = float(
            (order[:, :kk] == top1_b[:, None]).any(axis=1).mean()
        )
    ra = _rowrank(a)
    rb = _rowrank(b)
    out["spearman"] = float(np.mean([_pearson(x, y) for x, y in zip(ra, rb)]))
    return out

def _rowrank(X: np.ndarray) -> np.ndarray:
    order = np.argsort(X, axis=-1)
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(X.shape[0])[:, None]
    ranks[rows, order] = np.arange(X.shape[-1], dtype=np.float64)[None, :]
    return ranks

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    return float((xc * yc).sum() / den) if den > 0 else float("nan")
