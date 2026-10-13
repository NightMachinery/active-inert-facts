from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from linroute.backend import NUMPY, Backend, LinAlgError, as_host, get_backend

__all__ = [
    "Backend",
    "Head",
    "ProbeConfig",
    "Targets",
    "class_macro_accuracy",
    "decode",
    "fit_predict_affine",
    "feasible_q",
    "cv_folds",
    "stratified_folds",
    "lin_probe",
    "lin_probe_nested",
    "lin_probe_nested_batch",
    "lin_probe_biased_max",
    "lin_cls",
    "lin_cls_layer_max",
    "lin_cls_nested",
    "lin_cls_nested_batch",
    "lu_probe",
    "shuffle_control",
    "run_relation",
]

NAN = float("nan")

@dataclass(frozen=True)
class Head:
    U_cand: np.ndarray
    norm_gain: np.ndarray
    eps: float
    cand_token: np.ndarray

    @staticmethod
    def from_cache(cache: dict) -> "Head":
        h = cache["head"]
        return Head(
            U_cand=_f64(h["U_cand"]),
            norm_gain=_f64(h["norm_gain"]),
            eps=float(h["norm_eps"]),
            cand_token=_i64(cache["cand_first_token"]),
        )

    def decode(self, V: np.ndarray) -> np.ndarray:
        return decode(V, self.U_cand, self.norm_gain, self.eps)

    def tokens(self, cand_idx: np.ndarray) -> np.ndarray:
        return self.cand_token[cand_idx]

@dataclass(frozen=True)
class Targets:
    T: np.ndarray
    gold_token: np.ndarray
    label: np.ndarray

    def permuted(self, perm: np.ndarray) -> "Targets":
        return Targets(self.T[perm], self.gold_token[perm], self.label[perm])

@dataclass(frozen=True)
class ProbeConfig:
    q_primary: int = 16
    q_secondary: int | None = None
    outer_folds: int = 10
    inner_folds: int = 5
    folds: int = 5
    folds_seed: int = 0
    perm_seed: int = 0
    lam: float = 1.0
    n_perm: int = 5
    layers: tuple[int, ...] | None = None
    lu_lam: float = 1.0
    lu_folds: int = 5
    lu_train_frac: float = 0.75
    backend: str = "numpy"
    device: str = "cpu"
    dtype: str = "float64"

    @property
    def qs(self) -> tuple[int, ...]:
        qs = (self.q_primary,) if self.q_secondary is None else (
            self.q_primary, self.q_secondary
        )
        return tuple(dict.fromkeys(qs))

    def make_backend(self) -> Backend:
        return get_backend(self.backend, self.device, self.dtype)

def _f64(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x), dtype=np.float64)

def _i64(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(np.int64)

def _complement(n: int, test: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.arange(n), test)

def decode(
    V: np.ndarray,
    U_cand: np.ndarray,
    norm_gain: np.ndarray,
    eps: float,
) -> np.ndarray:
    v = torch.as_tensor(np.asarray(V), dtype=torch.float32)
    if v.ndim == 1:
        v = v[None, :]
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    g = torch.as_tensor(np.asarray(norm_gain), dtype=torch.float32)
    W = torch.as_tensor(np.asarray(U_cand), dtype=torch.float32)
    return ((g * v) @ W.T).argmax(-1).numpy().astype(np.int64)

BASIS_DRIVERS = ("eigh_gram", "gesdd", "svd_transposed")

basis_driver_counts: Counter = Counter()

def basis_driver_snapshot() -> dict[str, int]:
    return {k: int(basis_driver_counts.get(k, 0)) for k in BASIS_DRIVERS}

def basis_drivers_since(before: dict[str, int]) -> dict[str, int]:
    now = basis_driver_snapshot()
    return {k: now[k] - int(before.get(k, 0)) for k in BASIS_DRIVERS}

def basis_fallback_used(counts: dict[str, int]) -> bool:
    return any(int(counts.get(k, 0)) > 0 for k in BASIS_DRIVERS[1:])

def _svd_transposed(A, *, backend: Backend = NUMPY):
    P, s, _ = backend.svd(backend.transpose(A))
    return s, backend.transpose(P)

def _eigh_gram(A, *, backend: Backend = NUMPY):
    m, d = A.shape[-2], A.shape[-1]
    k = min(m, d)
    if m <= d:
        w, W = backend.eigh(backend.matmul(A, backend.transpose(A)))
        s = backend.sqrt(backend.clip(w[..., :k], 0.0, None))
        U = W[..., :, :k]
        tol = s[..., :1] * max(m, d) * backend.eps
        keep = s > tol
        Vt = backend.matmul(backend.transpose(U), A) / backend.where(keep, s, 1.0)[..., None]
        return s, backend.where(keep[..., None], Vt, 0.0)
    w, W = backend.eigh(backend.matmul(backend.transpose(A), A))
    s = backend.sqrt(backend.clip(w[..., :k], 0.0, None))
    return s, backend.transpose(W[..., :, :k])

def _basis_routes(A, *, backend: Backend):
    batch = A.shape[0] if A.ndim > 2 else None
    try:
        s, Vt = _eigh_gram(A, backend=backend)
        driver = "eigh_gram"
    except LinAlgError:
        if batch is not None and batch > 1:
            parts = [_basis_routes(A[i], backend=backend) for i in range(batch)]
            return (backend.stack([p[0] for p in parts]),
                    backend.stack([p[1] for p in parts]))
        try:
            _, s, Vt = backend.svd(A)
            driver = "gesdd"
        except LinAlgError:
            s, Vt = _svd_transposed(A, backend=backend)
            driver = "svd_transposed"
    basis_driver_counts[driver] += 1 if batch is None else batch
    return s, Vt

def _fit_basis(S_tr, *, backend: Backend = NUMPY):
    mu = backend.mean(S_tr, axis=-2)
    A = S_tr - mu[..., None, :]
    s, Vt = _basis_routes(A, backend=backend)
    return mu, Vt, s

def feasible_q(singular_values, n_rows: int, n_cols: int, q: int) -> bool:
    if q + 1 > n_rows:
        return False
    s = as_host(singular_values)
    if s.size == 0 or s[0] <= 0:
        return False
    tol = s[0] * max(n_rows, n_cols) * np.finfo(np.float64).eps
    return q <= int((s > tol).sum())

def _predict_with_basis(
    mu,
    Vt,
    S_tr,
    T_tr,
    S_te,
    q: int,
    lam: float,
    *,
    backend: Backend = NUMPY,
):
    B = backend
    V = B.transpose(Vt[..., :q, :])
    Z_tr = B.matmul(S_tr - mu[..., None, :], V)
    sd = B.std(Z_tr, axis=-2, keepdims=True) + 1e-8
    Z_tr = Z_tr / sd
    Z_te = B.matmul(S_te - mu[..., None, :], V) / sd
    t_mu = B.mean(T_tr, axis=-2)
    G = B.matmul(B.transpose(Z_tr), Z_tr) + lam * B.eye(Z_tr.shape[-1])
    W = B.solve(G, B.matmul(B.transpose(Z_tr), T_tr - t_mu))
    return B.matmul(Z_te, W) + t_mu, t_mu

def fit_predict_affine(
    S_tr,
    T_tr,
    S_te,
    q: int = 16,
    lam: float = 1.0,
    *,
    backend: Backend = NUMPY,
):
    mu, Vt, _ = _fit_basis(S_tr, backend=backend)
    return _predict_with_basis(mu, Vt, S_tr, T_tr, S_te, q, lam, backend=backend)

def _layer_stack(S, rows, layers, *, backend: Backend):
    x = S if rows is None else backend.take(S, rows, 0)
    return backend.contiguous(backend.swapaxes(backend.take(x, layers, 1), 0, 1))

def _select_batch(live: Sequence[int], n_layers: int, *arrays, backend: Backend):
    if len(live) == n_layers:
        return arrays
    idx = np.asarray(live, dtype=np.int64)
    return tuple(backend.take(a, idx, 0) for a in arrays)

def _decode_stack(head: "Head", Y, *, backend: Backend) -> np.ndarray:
    host = backend.to_numpy(Y)
    b, m, d = host.shape
    return head.decode(host.reshape(b * m, d)).reshape(b, m)

def cv_folds(n: int, folds: int, seed: int = 0) -> list[np.ndarray]:
    order = np.random.RandomState(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(order, folds) if len(f)]

def stratified_folds(labels: np.ndarray, folds: int, seed: int = 0) -> list[np.ndarray]:
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            buckets[j % folds].append(int(i))
    return [np.sort(np.asarray(b, dtype=np.int64)) for b in buckets if b]

def _nested_fold_plan(
    n: int,
    cfg: "ProbeConfig",
    folds: Sequence[Sequence[int]] | None = None,
    inner_folds: Sequence[Sequence[Sequence[int]]] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]]:
    if folds is None:
        if inner_folds is not None:
            raise ValueError("inner_folds was given without folds")
        plan = []
        for te in cv_folds(n, cfg.outer_folds, cfg.folds_seed):
            tr = _complement(n, te)
            plan.append(
                (te, tr, cv_folds(len(tr), cfg.inner_folds, cfg.folds_seed + 1))
            )
        return plan

    outer = [np.asarray(f, dtype=np.int64).ravel() for f in folds]
    if not outer:
        raise ValueError("folds is empty")
    flat = np.concatenate(outer) if outer else np.zeros(0, np.int64)
    if not np.array_equal(np.sort(flat), np.arange(n)):
        raise ValueError(
            f"the supplied outer folds do not partition the {n} subjects "
            f"({len(flat)} indices, {len(np.unique(flat))} distinct)"
        )
    if inner_folds is None:
        raise ValueError("explicit outer folds require explicit inner_folds")
    if len(inner_folds) != len(outer):
        raise ValueError(
            f"inner_folds has {len(inner_folds)} entries for {len(outer)} outer folds"
        )

    plan = []
    for te, inner in zip(outer, inner_folds):
        tr = _complement(n, te)
        where = np.full(n, -1, dtype=np.int64)
        where[tr] = np.arange(len(tr))
        rel: list[np.ndarray] = []
        for f in inner:
            idx = np.asarray(f, dtype=np.int64).ravel()
            local = where[idx]
            if (local < 0).any():
                raise ValueError(
                    "an inner validation fold contains subjects outside its "
                    "outer fitting partition"
                )
            rel.append(np.sort(local))
        seen = np.concatenate(rel) if rel else np.zeros(0, np.int64)
        if not np.array_equal(np.sort(seen), np.arange(len(tr))):
            raise ValueError(
                f"the inner folds do not partition their outer fitting "
                f"partition ({len(seen)} indices for {len(tr)} fitting rows)"
            )
        plan.append((np.sort(te), tr, rel))
    return plan

def _class_macro_mean(values: np.ndarray, gold: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    gold = np.asarray(gold)
    if values.shape[0] != len(gold):
        raise ValueError("values and gold must have the same subject axis")
    return float(np.mean([values[gold == c].mean() for c in np.unique(gold)]))

def class_macro_accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    return _class_macro_mean(np.asarray(pred) == np.asarray(gold), gold)

def _label_baselines(
    gold_tr: np.ndarray, gold_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values, counts = np.unique(gold_tr, return_counts=True)
    majority = values[counts.argmax()]
    freq = dict(zip(values.tolist(), (counts / counts.sum()).tolist()))
    return (
        (gold_te == majority),
        np.array([freq.get(int(g), 0.0) for g in gold_te], dtype=np.float64),
    )

def _obj_result(
    pred_cand: np.ndarray,
    base_cand: np.ndarray,
    majority_ok: np.ndarray,
    freq_p: np.ndarray,
    targets: Targets,
    head: Head,
    extra: dict,
) -> dict:
    gold = targets.gold_token
    pred_ok = head.tokens(pred_cand) == gold
    base_ok = head.tokens(base_cand) == gold
    acc = class_macro_accuracy(head.tokens(pred_cand), gold)
    base = class_macro_accuracy(head.tokens(base_cand), gold)
    return {
        "acc": acc,
        "base_acc": base,
        "delta": acc - base,
        "majority_acc": _class_macro_mean(majority_ok, gold),
        "freq_acc": _class_macro_mean(freq_p, gold),
        "micro_acc": float(pred_ok.mean()),
        "micro_base_acc": float(base_ok.mean()),
        "micro_delta": float(pred_ok.mean() - base_ok.mean()),
        "micro_majority_acc": float(majority_ok.mean()),
        "micro_freq_acc": float(freq_p.mean()),
        "n": int(len(gold)),
        "pred": pred_cand,
        "base_pred": base_cand,
        "available": True,
        **extra,
    }

def _unavailable(n: int, reason: str, extra: dict) -> dict:
    return {
        "acc": NAN,
        "base_acc": NAN,
        "delta": NAN,
        "majority_acc": NAN,
        "freq_acc": NAN,
        "micro_acc": NAN,
        "micro_base_acc": NAN,
        "micro_delta": NAN,
        "micro_majority_acc": NAN,
        "micro_freq_acc": NAN,
        "n": int(n),
        "pred": np.full(n, -1, dtype=np.int64),
        "base_pred": np.full(n, -1, dtype=np.int64),
        "available": False,
        "reason": reason,
        **extra,
    }

def _layer_slice(S: np.ndarray, layer: int | None) -> np.ndarray:
    if S.ndim == 2:
        return S
    if layer is None:
        raise ValueError("layer is required when S has a layer axis")
    return S[:, layer, :]

def lin_probe(
    S: np.ndarray,
    targets: Targets,
    head: Head,
    *,
    layer: int | None = None,
    q: int = 16,
    lam: float = 1.0,
    folds: int = 5,
    folds_seed: int = 0,
    backend: Backend = NUMPY,
) -> dict:
    X = _layer_slice(S, layer)
    n, d = X.shape
    extra = {"q": int(q), "layer": layer}
    pred = np.zeros(n, dtype=np.int64)
    base = np.zeros(n, dtype=np.int64)
    majority = np.zeros(n, dtype=bool)
    freq = np.zeros(n, dtype=np.float64)
    X_dev = backend.asarray(X)
    T_dev = backend.asarray(targets.T)

    for te in cv_folds(n, folds, folds_seed):
        tr = _complement(n, te)
        X_tr = backend.take(X_dev, tr, 0)
        mu, Vt, s = _fit_basis(X_tr, backend=backend)
        if not feasible_q(s, len(tr), d, q):
            return _unavailable(n, f"q={q} infeasible for {len(tr)} training rows", extra)
        T_hat, t_mu = _predict_with_basis(
            mu, Vt, X_tr, backend.take(T_dev, tr, 0), backend.take(X_dev, te, 0),
            q, lam, backend=backend,
        )
        t_mu = backend.to_numpy(t_mu)
        pred[te] = head.decode(backend.to_numpy(T_hat))
        base[te] = head.decode(np.repeat(t_mu[None], len(te), 0))
        majority[te], freq[te] = _label_baselines(
            targets.gold_token[tr], targets.gold_token[te]
        )
    return _obj_result(pred, base, majority, freq, targets, head, extra)

def lin_probe_nested_batch(
    S_all: np.ndarray,
    targets_list: Sequence[Targets],
    head: Head,
    cfg: ProbeConfig = ProbeConfig(),
    qs: Sequence[int] | None = None,
    *,
    folds: Sequence[Sequence[int]] | None = None,
    inner_folds: Sequence[Sequence[Sequence[int]]] | None = None,
) -> list[dict[int, dict]]:
    n, n_layer_axis, d = S_all.shape
    B = cfg.make_backend()
    qs = tuple(qs) if qs is not None else cfg.qs
    q_max = max(qs)
    layers = list(cfg.layers) if cfg.layers is not None else list(range(n_layer_axis))
    n_targets = len(targets_list)

    pred = {(i, q): np.zeros(n, np.int64) for i in range(n_targets) for q in qs}
    base = {(i, q): np.zeros(n, np.int64) for i in range(n_targets) for q in qs}
    majority = {(i, q): np.zeros(n, bool) for i in range(n_targets) for q in qs}
    freq = {(i, q): np.zeros(n, np.float64) for i in range(n_targets) for q in qs}
    chosen: dict[tuple[int, int], list[int]] = {
        (i, q): [] for i in range(n_targets) for q in qs
    }
    dead: set[tuple[int, int]] = set()

    S_dev = B.asarray(S_all)
    T_dev = [B.asarray(t.T) for t in targets_list]

    for te, tr, inner in _nested_fold_plan(n, cfg, folds, inner_folds):
        n_tr = len(tr)
        gold_tr = [t.gold_token[tr] for t in targets_list]
        T_tr = [B.take(T, tr, 0) for T in T_dev]

        ok = {(layer, q): True for layer in layers for q in qs}
        inner_pred = {
            (layer, i, q): np.zeros(n_tr, np.int64)
            for layer in layers for i in range(n_targets) for q in qs
        }

        X_tr = _layer_stack(S_dev, tr, layers, backend=B)
        X_te = _layer_stack(S_dev, te, layers, backend=B)
        mu_o, Vt_o, s_o = _fit_basis(X_tr, backend=B)
        Vt_o = Vt_o[..., :q_max, :]
        s_o_host = B.to_numpy(s_o)
        for li, layer in enumerate(layers):
            for q in qs:
                if not feasible_q(s_o_host[li], n_tr, d, q):
                    ok[(layer, q)] = False

        for te_in in inner:
            tr_in = _complement(n_tr, te_in)
            X_in, X_in_te = B.take(X_tr, tr_in, 1), B.take(X_tr, te_in, 1)
            T_in = [B.take(T, tr_in, 0) for T in T_tr]
            mu, Vt, s = _fit_basis(X_in, backend=B)
            s_host = B.to_numpy(s)
            for q in qs:
                for li, layer in enumerate(layers):
                    if ok[(layer, q)] and not feasible_q(s_host[li], len(tr_in), d, q):
                        ok[(layer, q)] = False

                live = [li for li, layer in enumerate(layers) if ok[(layer, q)]]
                if not live:
                    continue
                mu_l, Vt_l, X_l, X_te_l = _select_batch(
                    live, len(layers), mu, Vt, X_in, X_in_te, backend=B
                )
                for i in range(n_targets):
                    Y_hat, _ = _predict_with_basis(
                        mu_l, Vt_l, X_l, T_in[i], X_te_l, q, cfg.lam, backend=B
                    )
                    decoded = _decode_stack(head, Y_hat, backend=B)
                    for row, pos in enumerate(live):
                        inner_pred[(layers[pos], i, q)][te_in] = decoded[row]

        for i in range(n_targets):
            for q in qs:
                usable = [layer for layer in layers if ok[(layer, q)]]
                if not usable:
                    dead.add((i, q))
                    continue
                scores = [
                    class_macro_accuracy(
                        head.tokens(inner_pred[(layer, i, q)]), gold_tr[i]
                    )
                    for layer in usable
                ]
                best = usable[int(np.argmax(scores))]
                li = layers.index(best)
                Y_hat, t_mu = _predict_with_basis(
                    mu_o[li], Vt_o[li], X_tr[li], T_tr[i], X_te[li], q, cfg.lam,
                    backend=B,
                )
                t_mu = B.to_numpy(t_mu)
                pred[(i, q)][te] = head.decode(B.to_numpy(Y_hat))
                base[(i, q)][te] = head.decode(np.repeat(t_mu[None], len(te), 0))
                majority[(i, q)][te], freq[(i, q)][te] = _label_baselines(
                    gold_tr[i], targets_list[i].gold_token[te]
                )
                chosen[(i, q)].append(int(best))

    out: list[dict[int, dict]] = []
    for i, targets in enumerate(targets_list):
        per_q: dict[int, dict] = {}
        for q in qs:
            if (i, q) in dead:
                per_q[q] = _unavailable(n, f"q={q} infeasible at every layer", {"q": q})
                continue
            layers_chosen = chosen[(i, q)]
            per_q[q] = _obj_result(
                pred[(i, q)], base[(i, q)], majority[(i, q)], freq[(i, q)],
                targets, head,
                {
                    "q": int(q),
                    "layers": layers_chosen,
                    "modal_layer": int(np.bincount(layers_chosen).argmax()),
                },
            )
        out.append(per_q)
    return out

def lin_probe_nested(
    S_all: np.ndarray,
    targets: Targets,
    head: Head,
    cfg: ProbeConfig = ProbeConfig(),
    q: int | None = None,
    *,
    folds: Sequence[Sequence[int]] | None = None,
    inner_folds: Sequence[Sequence[Sequence[int]]] | None = None,
) -> dict:
    q = cfg.q_primary if q is None else q
    return lin_probe_nested_batch(
        S_all, [targets], head, cfg, (q,), folds=folds, inner_folds=inner_folds
    )[0][q]

def _layer_max_batch(
    S_all: np.ndarray,
    head: Head | None = None,
    *,
    qs: Sequence[int],
    lam: float = 1.0,
    folds: int = 5,
    folds_seed: int = 0,
    layers: Sequence[int] | None = None,
    cls_labels: Sequence[np.ndarray] = (),
    n_classes: int = 0,
    obj_targets: Sequence[Targets] = (),
    backend: Backend = NUMPY,
) -> tuple[list[dict[int, dict]], list[dict[int, dict]]]:
    n, _, d = S_all.shape
    cand = list(layers) if layers is not None else list(range(S_all.shape[1]))
    qs = tuple(dict.fromkeys(int(q) for q in qs))
    q_max = max(qs)
    fold_te = cv_folds(n, folds, folds_seed)

    Y_cls = []
    for lab in cls_labels:
        Y = np.zeros((n, n_classes), dtype=np.float64)
        Y[np.arange(n), lab] = 1.0
        Y_cls.append(Y)

    ok = {(layer, q): True for layer in cand for q in qs}
    cls_keys = [(j, layer, q) for layer in cand for q in qs for j in range(len(Y_cls))]
    obj_keys = [(i, layer, q) for layer in cand for q in qs for i in range(len(obj_targets))]
    cls_pred = {key: np.zeros(n, np.int64) for key in cls_keys}
    obj_pred = {key: np.zeros(n, np.int64) for key in obj_keys}
    obj_base = {key: np.zeros(n, np.int64) for key in obj_keys}
    obj_majority = {key: np.zeros(n, bool) for key in obj_keys}
    obj_freq = {key: np.zeros(n, np.float64) for key in obj_keys}

    S_dev = backend.asarray(S_all)
    Y_dev = [backend.asarray(Y) for Y in Y_cls]
    T_dev = [backend.asarray(tgt.T) for tgt in obj_targets]

    for te in fold_te:
        tr = _complement(n, te)

        X_tr = _layer_stack(S_dev, tr, cand, backend=backend)
        X_te = _layer_stack(S_dev, te, cand, backend=backend)
        mu, Vt, s = _fit_basis(X_tr, backend=backend)
        Vt_q = Vt[..., :q_max, :]
        s_host = backend.to_numpy(s)
        Y_tr = [backend.take(Y, tr, 0) for Y in Y_dev]
        T_tr = [backend.take(T, tr, 0) for T in T_dev]
        for q in qs:
            for li, layer in enumerate(cand):
                if ok[(layer, q)] and not feasible_q(s_host[li], len(tr), d, q):
                    ok[(layer, q)] = False
            live = [li for li, layer in enumerate(cand) if ok[(layer, q)]]
            if not live:
                continue
            mu_l, Vt_l, X_l, X_te_l = _select_batch(
                live, len(cand), mu, Vt_q, X_tr, X_te, backend=backend
            )
            for j in range(len(Y_dev)):
                scores, _ = _predict_with_basis(
                    mu_l, Vt_l, X_l, Y_tr[j], X_te_l, q, lam, backend=backend
                )

                chosen_cls = backend.to_numpy(scores).argmax(-1)
                for row, pos in enumerate(live):
                    cls_pred[(j, cand[pos], q)][te] = chosen_cls[row]
            for i, tgt in enumerate(obj_targets):
                T_hat, t_mu = _predict_with_basis(
                    mu_l, Vt_l, X_l, T_tr[i], X_te_l, q, lam, backend=backend
                )
                decoded = _decode_stack(head, T_hat, backend=backend)

                t_mu = backend.to_numpy(t_mu)
                base_pred = head.decode(np.repeat(t_mu[None], len(te), 0))
                majority_ok, freq_p = _label_baselines(
                    tgt.gold_token[tr], tgt.gold_token[te]
                )
                for row, pos in enumerate(live):
                    key = (i, cand[pos], q)
                    obj_pred[key][te] = decoded[row]
                    obj_base[key][te] = base_pred
                    obj_majority[key][te] = majority_ok
                    obj_freq[key][te] = freq_p

    cls_out: list[dict[int, dict]] = []
    for j, lab in enumerate(cls_labels):
        counts = np.bincount(lab, minlength=n_classes)
        per_q: dict[int, dict] = {}
        for q in qs:
            usable = [layer for layer in cand if ok[(layer, q)]]
            if not usable:
                per_q[q] = {"acc": NAN, "micro_acc": NAN, "majority_share": NAN,
                            "chance": 1.0 / n_classes, "q": int(q), "layer": None,
                            "n": int(n), "available": False, "biased": True}
                continue
            accs = [class_macro_accuracy(cls_pred[(j, layer, q)], lab) for layer in usable]
            best = int(np.argmax(accs))
            per_q[q] = {
                "acc": accs[best],
                "micro_acc": float((cls_pred[(j, usable[best], q)] == lab).mean()),
                "majority_share": float(counts.max() / n),
                "chance": 1.0 / float(n_classes),
                "q": int(q),
                "layer": usable[best],
                "n": int(n),
                "available": True,
                "biased": True,
            }
        cls_out.append(per_q)

    obj_out: list[dict[int, dict]] = []
    for i, tgt in enumerate(obj_targets):
        per_q = {}
        for q in qs:
            per_layer = [
                _obj_result(
                    obj_pred[(i, layer, q)], obj_base[(i, layer, q)],
                    obj_majority[(i, layer, q)], obj_freq[(i, layer, q)],
                    tgt, head, {"q": int(q), "layer": layer},
                ) if ok[(layer, q)] else None
                for layer in cand
            ]
            usable = [r for r in per_layer if r is not None]
            if not usable:
                per_q[q] = _unavailable(
                    len(tgt.gold_token), f"q={q} infeasible", {"q": q}
                )
                continue
            best = max(usable, key=lambda r: r["delta"])
            per_q[q] = {**best, "biased": True, "delta_by_layer": [
                float(r["delta"]) if r is not None else NAN for r in per_layer
            ]}
        obj_out.append(per_q)

    return cls_out, obj_out

def lin_probe_biased_max(
    S_all: np.ndarray,
    targets: Targets,
    head: Head,
    *,
    q: int = 16,
    lam: float = 1.0,
    folds: int = 5,
    folds_seed: int = 0,
    layers: Sequence[int] | None = None,
    backend: Backend = NUMPY,
) -> dict:
    _, obj = _layer_max_batch(
        S_all, head, qs=(q,), lam=lam, folds=folds, folds_seed=folds_seed,
        layers=layers, obj_targets=(targets,), backend=backend,
    )
    return obj[0][int(q)]

def lin_cls(
    S: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    layer: int | None = None,
    q: int = 16,
    lam: float = 1.0,
    folds: int = 5,
    folds_seed: int = 0,
    backend: Backend = NUMPY,
) -> dict:
    X = _layer_slice(S, layer)
    n, d = X.shape
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), labels] = 1.0

    pred = np.zeros(n, dtype=np.int64)
    X_dev = backend.asarray(X)
    Y_dev = backend.asarray(Y)
    for te in cv_folds(n, folds, folds_seed):
        tr = _complement(n, te)
        X_tr = backend.take(X_dev, tr, 0)
        mu, Vt, s = _fit_basis(X_tr, backend=backend)
        if not feasible_q(s, len(tr), d, q):
            return {"acc": NAN, "micro_acc": NAN, "majority_share": NAN,
                    "chance": 1.0 / n_classes,
                    "q": int(q), "layer": layer, "n": int(n), "available": False}
        scores, _ = _predict_with_basis(
            mu, Vt, X_tr, backend.take(Y_dev, tr, 0), backend.take(X_dev, te, 0),
            q, lam, backend=backend,
        )
        pred[te] = backend.to_numpy(scores).argmax(-1)

    counts = np.bincount(labels, minlength=n_classes)
    return {
        "acc": class_macro_accuracy(pred, labels),
        "micro_acc": float((pred == labels).mean()),
        "majority_share": float(counts.max() / n),
        "chance": 1.0 / float(n_classes),
        "q": int(q),
        "layer": layer,
        "n": int(n),
        "available": True,
    }

def lin_cls_layer_max(
    S_all: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    q: int = 16,
    lam: float = 1.0,
    folds: int = 5,
    folds_seed: int = 0,
    layers: Sequence[int] | None = None,
    backend: Backend = NUMPY,
) -> dict:
    cls, _ = _layer_max_batch(
        S_all, qs=(q,), lam=lam, folds=folds, folds_seed=folds_seed,
        layers=layers, cls_labels=(labels,), n_classes=n_classes,
        backend=backend,
    )
    return cls[0][int(q)]

def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((len(labels), n_classes), dtype=np.float64)
    Y[np.arange(len(labels)), labels] = 1.0
    return Y

def _cls_unavailable(n: int, n_classes: int, q: int, reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "acc": NAN,
        "macro_acc": NAN,
        "micro_acc": NAN,
        "majority_acc": NAN,
        "micro_majority_acc": NAN,
        "gain": NAN,
        "micro_gain": NAN,
        "majority_share": NAN,
        "chance": 1.0 / float(n_classes),
        "q": int(q),
        "n": int(n),
        "n_classes": int(n_classes),
        "n_classes_present": 0,
        "layers": [],
        "modal_layer": None,
        "pred": np.full(n, -1, dtype=np.int64),
    }

def lin_cls_nested_batch(
    S_all: np.ndarray,
    label_sets: Sequence[np.ndarray],
    n_classes: int,
    cfg: ProbeConfig = ProbeConfig(),
    qs: Sequence[int] | None = None,
    *,
    folds: Sequence[Sequence[int]] | None = None,
    inner_folds: Sequence[Sequence[Sequence[int]]] | None = None,
) -> list[dict[int, dict]]:
    n, n_layer_axis, d = S_all.shape
    B = cfg.make_backend()
    qs = tuple(dict.fromkeys(int(q) for q in (qs if qs is not None else cfg.qs)))
    q_max = max(qs)
    layers = list(cfg.layers) if cfg.layers is not None else list(range(n_layer_axis))
    label_sets = [_i64(lab) for lab in label_sets]
    n_sets = len(label_sets)
    for lab in label_sets:
        if len(lab) != n:
            raise ValueError("a label set does not match the subject axis")
        if lab.size and (int(lab.min()) < 0 or int(lab.max()) >= n_classes):
            raise ValueError(
                f"labels must be effective classes in 0..{n_classes - 1}; got "
                f"{int(lab.min())}..{int(lab.max())}"
            )

    pred = {(j, q): np.zeros(n, np.int64) for j in range(n_sets) for q in qs}
    majority = {(j, q): np.zeros(n, bool) for j in range(n_sets) for q in qs}
    chosen: dict[tuple[int, int], list[int]] = {
        (j, q): [] for j in range(n_sets) for q in qs
    }
    dead: set[tuple[int, int]] = set()

    S_dev = B.asarray(S_all)

    for te, tr, inner in _nested_fold_plan(n, cfg, folds, inner_folds):
        n_tr = len(tr)
        lab_tr = [lab[tr] for lab in label_sets]
        Y_tr = [B.asarray(_one_hot(lab, n_classes)) for lab in lab_tr]

        ok = {(layer, q): True for layer in layers for q in qs}
        inner_pred = {
            (layer, j, q): np.zeros(n_tr, np.int64)
            for layer in layers for j in range(n_sets) for q in qs
        }

        X_tr = _layer_stack(S_dev, tr, layers, backend=B)
        X_te = _layer_stack(S_dev, te, layers, backend=B)
        mu_o, Vt_o, s_o = _fit_basis(X_tr, backend=B)
        Vt_o = Vt_o[..., :q_max, :]
        s_o_host = B.to_numpy(s_o)
        for li, layer in enumerate(layers):
            for q in qs:
                if not feasible_q(s_o_host[li], n_tr, d, q):
                    ok[(layer, q)] = False

        for te_in in inner:
            tr_in = _complement(n_tr, te_in)
            X_in, X_in_te = B.take(X_tr, tr_in, 1), B.take(X_tr, te_in, 1)
            Y_in = [B.take(Y, tr_in, 0) for Y in Y_tr]
            mu, Vt, s = _fit_basis(X_in, backend=B)
            s_host = B.to_numpy(s)
            for q in qs:
                for li, layer in enumerate(layers):
                    if ok[(layer, q)] and not feasible_q(s_host[li], len(tr_in), d, q):
                        ok[(layer, q)] = False
                live = [li for li, layer in enumerate(layers) if ok[(layer, q)]]
                if not live:
                    continue
                mu_l, Vt_l, X_l, X_te_l = _select_batch(
                    live, len(layers), mu, Vt, X_in, X_in_te, backend=B
                )
                for j in range(n_sets):
                    scores, _ = _predict_with_basis(
                        mu_l, Vt_l, X_l, Y_in[j], X_te_l, q, cfg.lam, backend=B
                    )
                    chosen_cls = B.to_numpy(scores).argmax(-1)
                    for row, pos in enumerate(live):
                        inner_pred[(layers[pos], j, q)][te_in] = chosen_cls[row]

        for j in range(n_sets):
            for q in qs:
                usable = [layer for layer in layers if ok[(layer, q)]]
                if not usable:
                    dead.add((j, q))
                    continue
                scores_by_layer = [
                    class_macro_accuracy(inner_pred[(layer, j, q)], lab_tr[j])
                    for layer in usable
                ]
                best = usable[int(np.argmax(scores_by_layer))]
                li = layers.index(best)
                scores, _ = _predict_with_basis(
                    mu_o[li], Vt_o[li], X_tr[li], Y_tr[j], X_te[li], q, cfg.lam,
                    backend=B,
                )
                pred[(j, q)][te] = B.to_numpy(scores).argmax(-1)
                counts_tr = np.bincount(lab_tr[j], minlength=n_classes)
                majority[(j, q)][te] = label_sets[j][te] == int(counts_tr.argmax())
                chosen[(j, q)].append(int(best))

    out: list[dict[int, dict]] = []
    for j, lab in enumerate(label_sets):
        counts = np.bincount(lab, minlength=n_classes)
        per_q: dict[int, dict] = {}
        for q in qs:
            if (j, q) in dead:
                per_q[q] = _cls_unavailable(
                    n, n_classes, q, f"q={q} infeasible at every layer"
                )
                continue
            p = pred[(j, q)]
            acc = class_macro_accuracy(p, lab)
            maj = _class_macro_mean(majority[(j, q)], lab)
            per_q[q] = {
                "available": True,
                "acc": acc,
                "macro_acc": acc,
                "micro_acc": float((p == lab).mean()),
                "majority_acc": maj,
                "micro_majority_acc": float(majority[(j, q)].mean()),
                "gain": acc - maj,
                "micro_gain": float((p == lab).mean() - majority[(j, q)].mean()),
                "majority_share": float(counts.max() / n),
                "chance": 1.0 / float(n_classes),
                "q": int(q),
                "n": int(n),
                "n_classes": int(n_classes),
                "n_classes_present": int((counts > 0).sum()),
                "layers": chosen[(j, q)],
                "modal_layer": int(np.bincount(chosen[(j, q)]).argmax()),
                "pred": p,
            }
        out.append(per_q)
    return out

def lin_cls_nested(
    S_all: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    cfg: ProbeConfig = ProbeConfig(),
    q: int | None = None,
    *,
    folds: Sequence[Sequence[int]] | None = None,
    inner_folds: Sequence[Sequence[Sequence[int]]] | None = None,
) -> dict:
    q = cfg.q_primary if q is None else q
    return lin_cls_nested_batch(
        S_all, [labels], n_classes, cfg, (q,),
        folds=folds, inner_folds=inner_folds,
    )[0][q]

def row_cosine(A, B, eps: float = 1e-12, *, backend: Backend = NUMPY):
    num = backend.sum(A * B, axis=-1)
    den = backend.norm(A, axis=-1) * backend.norm(B, axis=-1) + eps
    return num / den

def ridge_affine_predict(
    X_tr, Y_tr, X_te, lam: float = 1.0, *, backend: Backend = NUMPY
):
    mu_x = backend.mean(X_tr, axis=-2)
    mu_y = backend.mean(Y_tr, axis=-2)
    Xc = X_tr - mu_x
    Yc = Y_tr - mu_y
    G = backend.matmul(Xc, backend.transpose(Xc))
    if lam > 0:
        G = G + lam * backend.eye(Xc.shape[-2])
    alpha = backend.solve(G, Yc)
    return backend.matmul(
        backend.matmul(X_te - mu_x, backend.transpose(Xc)), alpha
    ) + mu_y

def _lu_scores(
    S_tr,
    O_tr,
    S_te,
    O_te,
    lam: float,
    *,
    backend: Backend = NUMPY,
) -> dict:
    direction = backend.mean(O_tr - S_tr, axis=-2)
    const = backend.broadcast_to(
        backend.mean(O_tr, axis=-2)[None], (O_te.shape[0], O_te.shape[1])
    )
    return {
        "base": row_cosine(S_te, O_te, backend=backend),
        "affine": row_cosine(
            ridge_affine_predict(S_tr, O_tr, S_te, lam, backend=backend), O_te,
            backend=backend,
        ),
        "translation": row_cosine(S_te + direction, O_te, backend=backend),
        "const": row_cosine(const, O_te, backend=backend),
    }

def _lu_split(n: int, train_frac: float, seed: int) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_train = max(8, int(train_frac * n))
    n_train = min(n_train, max(1, n - 4))
    mask = np.zeros(n, dtype=bool)
    mask[perm[:n_train]] = True
    return mask

def lu_probe(
    S,
    O,
    *,
    folds: int = 5,
    folds_seed: int = 0,
    lam: float = 1.0,
    train_frac: float = 0.75,
    backend: Backend = NUMPY,
) -> dict:
    n = S.shape[0]
    keys = ("base", "affine", "translation", "const")
    S_dev, O_dev = backend.asarray(S), backend.asarray(O)

    def rows(idx):
        return backend.take(S_dev, idx, 0), backend.take(O_dev, idx, 0)

    pooled = {key: np.zeros(n) for key in keys}
    for te in cv_folds(n, folds, folds_seed):
        tr = _complement(n, te)
        S_tr, O_tr = rows(tr)
        S_te, O_te = rows(te)
        scored = _lu_scores(S_tr, O_tr, S_te, O_te, lam, backend=backend)
        for key, val in scored.items():
            pooled[key][te] = backend.to_numpy(val)

    mask = _lu_split(n, train_frac, folds_seed)
    S_tr, O_tr = rows(np.flatnonzero(mask))
    S_te, O_te = rows(np.flatnonzero(~mask))
    single = _lu_scores(S_tr, O_tr, S_te, O_te, lam, backend=backend)

    out: dict = {"n": int(n), "n_test_split": int((~mask).sum())}
    for key in keys:
        out[f"{key}_cos_cv"] = float(pooled[key].mean())
        out[f"{key}_cos_split"] = float(backend.to_numpy(single[key]).mean())
    for variant in ("affine", "translation", "const"):
        out[f"delta_cos_{variant}_cv"] = out[f"{variant}_cos_cv"] - out["base_cos_cv"]
        out[f"delta_cos_{variant}_split"] = (
            out[f"{variant}_cos_split"] - out["base_cos_split"]
        )
    out["delta_cos_cv"] = out["delta_cos_affine_cv"]
    out["delta_cos_split"] = out["delta_cos_affine_split"]
    return out

def permutations(n: int, n_perm: int, seed: int) -> list[np.ndarray]:
    return [
        np.random.RandomState(seed + 1000 + i).permutation(n) for i in range(n_perm)
    ]

def shuffle_control(
    score_fn: Callable[..., float],
    targets: Sequence[np.ndarray],
    n_perm: int = 5,
    seed: int = 0,
) -> dict:
    n = len(targets[0])
    vals = [
        float(score_fn(*[np.asarray(t)[p] for t in targets]))
        for p in permutations(n, n_perm, seed)
    ]
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "values": vals,
    }

def _confounds(labels: np.ndarray, n_cand: int, cand_token: np.ndarray) -> dict:
    n = len(labels)
    counts = np.bincount(labels, minlength=n_cand).astype(np.float64)
    p = counts[counts > 0] / n
    entropy = float(-(p * np.log(p)).sum())
    n_distinct_tokens = int(len(np.unique(cand_token)))
    return {
        "n": int(n),
        "n_objects": int(n_cand),
        "n_objects_present": int((counts > 0).sum()),
        "k_per_object": float(n / n_cand),
        "top1_share": float(counts.max() / n),
        "entropy": entropy,
        "entropy_norm": float(entropy / np.log(n_cand)) if n_cand > 1 else 0.0,
        "cand_token_distinct": n_distinct_tokens,
        "cand_token_collision_rate": float(1.0 - n_distinct_tokens / n_cand),
    }

def _obj_probe_block(
    name: str,
    S_all: np.ndarray,
    targets: Targets,
    head: Head,
    cfg: ProbeConfig,
    head_pred: np.ndarray,
    biased: dict[int, dict],
) -> dict:
    perms = permutations(len(targets.gold_token), cfg.n_perm, cfg.perm_seed)
    batch = [targets] + [targets.permuted(p) for p in perms]
    results = lin_probe_nested_batch(S_all, batch, head, cfg)

    out: dict = {}
    for q in cfg.qs:
        real = results[0][q]
        shuffled = [r[q]["delta"] for r in results[1:]]
        prefix = f"{name}_q{q}"
        out[f"{prefix}_available"] = real["available"]
        out[f"{prefix}_acc"] = real["acc"]
        out[f"{prefix}_base_acc"] = real["base_acc"]
        out[f"{prefix}_delta"] = real["delta"]
        out[f"{prefix}_majority_acc"] = real["majority_acc"]
        out[f"{prefix}_freq_acc"] = real["freq_acc"]
        out[f"{prefix}_layers"] = real.get("layers")
        out[f"{prefix}_modal_layer"] = real.get("modal_layer")
        out[f"{prefix}_agree_head"] = (
            _class_macro_mean(
                head.tokens(real["pred"]) == head.tokens(head_pred), targets.gold_token
            )
            if real["available"] else NAN
        )
        out[f"{prefix}_micro_agree_head"] = (
            float((head.tokens(real["pred"]) == head.tokens(head_pred)).mean())
            if real["available"] else NAN
        )
        out[f"{prefix}_shuffle_delta_mean"] = float(np.mean(shuffled))
        out[f"{prefix}_shuffle_delta_std"] = float(np.std(shuffled))

        out[f"{prefix}_layer_max_biased_delta"] = biased[q]["delta"]
        out[f"{prefix}_layer_max_biased_layer"] = biased[q].get("layer")
    return out

def run_relation(cache: dict, cfg: ProbeConfig = ProbeConfig()) -> dict:
    meta = cache["meta"]
    head = Head.from_cache(cache)
    labels = _i64(cache["cand_index"])
    n_cand = len(cache["candidates"])

    has_lu = "rel_gold_object_mean" in cache
    has_fs = "fs_correct" in cache

    rendered_token = _i64(cache["object_first_token"])
    gold_token = head.cand_token[labels]

    rel_subject_last = _f64(cache["rel_subject_last"]).mean(1)
    neutral_subject_last = _f64(cache["neutral_subject_last"]).mean(1)
    rel_answer_final = _f64(cache["rel_answer_final"]).mean(1)
    layer_axis = rel_subject_last.shape[1]
    L = layer_axis - 1

    lu_subject_layer = L // 2 + 1
    lu_object_layer = max(0, L - 1)
    rel_subject_mean = _f64(cache["rel_subject_mean"][:, :, lu_subject_layer, :]).mean(1)
    neutral_subject_mean = _f64(cache["neutral_subject_mean"][:, :, lu_subject_layer, :]).mean(1)
    rel_gold_object_mean = (
        _f64(cache["rel_gold_object_mean"][:, :, lu_object_layer, :]).mean(1)
        if has_lu else None
    )

    _, effective_labels = np.unique(gold_token, return_inverse=True)
    n_effective_classes = int(effective_labels.max()) + 1
    targets = Targets(T=rel_answer_final, gold_token=gold_token, label=effective_labels)

    out: dict = {
        "model": meta["model"],
        "relation": meta["relation"],
        "relation_type": meta.get("relation_type"),
        "n_layers": int(meta.get("n_layers", L)),
        "layer_axis": int(layer_axis),
        "q_primary": int(cfg.q_primary),
        "q_secondary": cfg.q_secondary,
        "folds_seed": int(cfg.folds_seed),
        "n_perm": int(cfg.n_perm),
        "lu_subject_layer": int(lu_subject_layer),
        "lu_object_layer": int(lu_object_layer),
        "lu_available": bool(has_lu),
        "epoch": int(meta["epoch"]) if "epoch" in meta else None,
        "arm": meta.get("arm"),
        "first_token_mismatch_rate": float((rendered_token != gold_token).mean()),
    }
    out.update(_confounds(labels, n_cand, head.cand_token))
    for key in ("cand_collision_rate", "cand_token_collision_rate"):
        if key in meta and meta[key] is not None:
            out["cand_token_collision_rate_meta"] = float(meta[key])
            break
    if meta.get("first_token_mismatch_rate") is not None:
        out["first_token_mismatch_rate_meta"] = float(meta["first_token_mismatch_rate"])

    zs_cand = _i64(cache["zs_cand"])
    zs_hit = head.tokens(zs_cand) == gold_token[:, None]
    out["zs_acc"] = _class_macro_mean(zs_hit, gold_token)
    out["zs_acc_t0"] = _class_macro_mean(zs_hit[:, 0], gold_token)
    out["zs_micro_acc"] = float(zs_hit.mean())
    out["zs_micro_acc_t0"] = float(zs_hit[:, 0].mean())
    out["zs_acc_index"] = float((zs_cand == labels[:, None]).mean())
    out["fs_acc"] = (
        _class_macro_mean(_i64(cache["fs_correct"]), gold_token) if has_fs else NAN
    )
    out["fs_micro_acc"] = float(_i64(cache["fs_correct"]).mean()) if has_fs else NAN
    head_pred = head.decode(rel_answer_final)
    out["head_acc"] = class_macro_accuracy(head.tokens(head_pred), gold_token)
    out["head_micro_acc"] = float((head.tokens(head_pred) == gold_token).mean())

    perms = permutations(len(labels), cfg.n_perm, cfg.perm_seed)
    backend = cfg.make_backend()
    layer_max = {
        block: _layer_max_batch(
            states, head, qs=cfg.qs, lam=cfg.lam, folds=cfg.folds,
            folds_seed=cfg.folds_seed, layers=cfg.layers,
            cls_labels=[effective_labels] + [effective_labels[p] for p in perms],
            n_classes=n_effective_classes,
            obj_targets=(targets,), backend=backend,
        )
        for block, states in (("rel", rel_subject_last), ("neu", neutral_subject_last))
    }

    out.update(_obj_probe_block("lin_obj", rel_subject_last, targets, head, cfg,
                                head_pred, layer_max["rel"][1][0]))
    out.update(_obj_probe_block("lin_neu_obj", neutral_subject_last, targets, head, cfg,
                                head_pred, layer_max["neu"][1][0]))

    for name, block in (("lin_cls_rel", "rel"), ("lin_cls_neu", "neu")):
        cls = layer_max[block][0]
        for q in cfg.qs:
            res = cls[0][q]
            shuffled = [r[q]["acc"] for r in cls[1:]]
            prefix = f"{name}_q{q}"
            out[f"{prefix}_acc"] = res["acc"]
            out[f"{prefix}_micro_acc"] = res["micro_acc"]
            out[f"{prefix}_layer"] = res["layer"]
            out[f"{prefix}_available"] = res["available"]
            out[f"{prefix}_shuffle_acc_mean"] = float(np.mean(shuffled))
            out[f"{prefix}_shuffle_acc_std"] = float(np.std(shuffled))
    out["lin_cls_majority"] = float(
        np.bincount(effective_labels, minlength=n_effective_classes).max() / len(labels)
    )
    out["lin_cls_chance"] = 1.0 / float(n_effective_classes)

    lu_blocks = (
        (("lu_obj", rel_subject_mean), ("lu_neu_obj", neutral_subject_mean))
        if has_lu
        else ()
    )
    for name, S in lu_blocks:
        res = lu_probe(S, rel_gold_object_mean, folds=cfg.lu_folds, folds_seed=cfg.folds_seed,
                       lam=cfg.lu_lam, train_frac=cfg.lu_train_frac, backend=backend)
        for key, val in res.items():
            out[f"{name}_{key}"] = val
        control = shuffle_control(
            lambda O_p, _S=S: lu_probe(
                _S, O_p, folds=cfg.lu_folds, folds_seed=cfg.folds_seed,
                lam=cfg.lu_lam, train_frac=cfg.lu_train_frac, backend=backend,
            )["delta_cos_cv"],
            (rel_gold_object_mean,), n_perm=cfg.n_perm, seed=cfg.perm_seed,
        )
        out[f"{name}_shuffle_delta_cos_mean"] = control["mean"]
        out[f"{name}_shuffle_delta_cos_std"] = control["std"]

    return out
