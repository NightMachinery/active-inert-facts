from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from linroute.probes import (
    Head,
    ProbeConfig,
    Targets,
    _complement,
    _label_baselines,
    _nested_fold_plan,
    _obj_result,
    _unavailable,
    class_macro_accuracy,
)

NAN = float("nan")

BETA_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)

LAYER_GRID_D1: tuple[int, ...] = (8, 14, 20, 26, 31)

def layer_grid(n_layers: int, size: int = 5) -> tuple[int, ...]:
    if size < 2:
        raise ValueError("a grid needs at least two layers")
    lo, hi = 1, n_layers - 1
    if hi < lo:
        raise ValueError(f"n_layers={n_layers} is too small for a grid")
    raw = np.linspace(lo, hi, size)

    return tuple(sorted({int(np.floor(x + 0.5)) for x in raw}))

def lre_predict(
    G: np.ndarray,
    T: np.ndarray,
    fit: np.ndarray,
    test: np.ndarray,
    beta: float,
) -> np.ndarray:
    fit = np.asarray(fit, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    W_s_test = G[np.ix_(fit, test)].mean(axis=0)
    W_s_mean = G[np.ix_(fit, fit)].mean(axis=(0, 1))
    b = T[fit].mean(axis=0) - W_s_mean
    return beta * W_s_test + b

@dataclass(frozen=True)
class LreInputs:
    G_by_layer: dict[int, np.ndarray]
    T: np.ndarray

    def __post_init__(self) -> None:
        n = self.T.shape[0]
        for layer, G in self.G_by_layer.items():
            if G.shape[:2] != (n, n):
                raise ValueError(
                    f"layer {layer}: G has shape {G.shape}, expected ({n}, {n}, d)"
                )

def lin_lre_nested(
    inputs: LreInputs,
    targets: Targets,
    head: Head,
    cfg: ProbeConfig = ProbeConfig(),
    betas: Sequence[float] = BETA_GRID,
    *,
    folds: Sequence[Sequence[int]] | None = None,
    inner_folds: Sequence[Sequence[Sequence[int]]] | None = None,
) -> dict:
    n = len(targets.gold_token)
    layers = sorted(inputs.G_by_layer)
    betas = tuple(float(b) for b in betas)
    if not layers:
        return _unavailable(n, "no layer in the declared grid", {"q": None})

    pred = np.zeros(n, np.int64)
    base = np.zeros(n, np.int64)
    majority = np.zeros(n, bool)
    freq = np.zeros(n, np.float64)
    chosen_layer: list[int] = []
    chosen_beta: list[float] = []

    for te, tr, inner in _nested_fold_plan(n, cfg, folds, inner_folds):
        n_tr = len(tr)
        gold_tr = targets.gold_token[tr]

        inner_pred = {
            (layer, b): np.zeros(n_tr, np.int64) for layer in layers for b in betas
        }
        for layer in layers:
            G = inputs.G_by_layer[layer]
            for te_in in inner:
                tr_in = _complement(n_tr, te_in)
                fit = tr[tr_in]
                test = tr[te_in]
                for b in betas:
                    Y_hat = lre_predict(G, inputs.T, fit, test, b)
                    inner_pred[(layer, b)][te_in] = head.decode(Y_hat)

        combos = [(layer, b) for layer in layers for b in betas]
        scores = [
            class_macro_accuracy(head.tokens(inner_pred[c]), gold_tr) for c in combos
        ]
        best_layer, best_beta = combos[int(np.argmax(scores))]

        Y_hat = lre_predict(inputs.G_by_layer[best_layer], inputs.T, tr, te, best_beta)
        pred[te] = head.decode(Y_hat)
        t_mu = inputs.T[tr].mean(axis=0)
        base[te] = head.decode(np.repeat(t_mu[None], len(te), 0))
        majority[te], freq[te] = _label_baselines(gold_tr, targets.gold_token[te])
        chosen_layer.append(int(best_layer))
        chosen_beta.append(float(best_beta))

    return _obj_result(
        pred,
        base,
        majority,
        freq,
        targets,
        head,
        {
            "q": None,
            "layers": chosen_layer,
            "betas": chosen_beta,
            "modal_layer": int(np.bincount(np.asarray(chosen_layer)).argmax()),
            "modal_beta": float(
                max(set(chosen_beta), key=chosen_beta.count)
            ),
            "layer_grid": tuple(layers),
            "beta_grid": betas,
        },
    )

def direction_table_from_jacobians(J: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.einsum("vij,uj->vui", J, S, optimize=True)

def jvp_direction_table(
    fn,
    states: np.ndarray,
    directions: np.ndarray | None = None,
    chunk: int = 64,
) -> np.ndarray:
    import torch
    from torch.func import jvp, vmap

    fns = list(fn) if isinstance(fn, (list, tuple)) else None
    n, d = states.shape
    dirs = states if directions is None else directions
    m = dirs.shape[0]
    out = np.zeros((n, m, d), dtype=np.float32)
    S = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    D = torch.as_tensor(np.asarray(dirs), dtype=torch.float32)

    for v in range(n):
        f = fns[v] if fns is not None else fn
        s_v = S[v]

        def one(tangent, _f=f, _s=s_v):
            return jvp(_f, (_s,), (tangent,))[1]

        for lo in range(0, m, chunk):
            hi = min(lo + chunk, m)
            out[v, lo:hi] = vmap(one)(D[lo:hi]).detach().cpu().numpy()
    return out

def jvp_flops(params_above: float, n_suffix: int) -> float:
    return 2.0 * params_above * n_suffix

def direction_table_cost(
    n_subjects: int,
    n_layers_in_grid: int,
    params_above: float,
    n_suffix: int,
    d_model: int = 4096,
    tflops: float = 80.0,
) -> dict:
    per = jvp_flops(params_above, n_suffix)
    jvps_directed = n_subjects * n_subjects * n_layers_in_grid
    jvps_dense = n_subjects * d_model * n_layers_in_grid
    rate = tflops * 1e12
    return {
        "n_jvp_directed": jvps_directed,
        "n_jvp_dense": jvps_dense,
        "flops_per_jvp": per,
        "seconds_directed": jvps_directed * per / rate,
        "seconds_dense": jvps_dense * per / rate,
        "speedup": d_model / max(n_subjects, 1),
        "bytes_table": n_subjects * n_subjects * d_model * 4 * n_layers_in_grid,
        "bytes_dense_accumulators": 61 * d_model * d_model * 4 * n_layers_in_grid,
    }

def checkpoint_bytes(n_params: float, dtype_bytes: int = 2) -> float:
    return n_params * dtype_bytes
