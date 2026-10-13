from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class Design:
    X: np.ndarray
    start: np.ndarray
    chosen: np.ndarray
    cell: np.ndarray
    names: list[str]

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(np.append(self.start, len(self.X)))

    def columns(self, keep: list[str]) -> "Design":
        ix = [self.names.index(k) for k in keep]
        return Design(self.X[:, ix], self.start, self.chosen, self.cell, list(keep))

def _probs(D: Design, beta: np.ndarray):
    u = D.X @ beta
    seg = np.repeat(np.arange(len(D.start)), D.sizes)
    m = np.maximum.reduceat(u, D.start)
    e = np.exp(u - m[seg])
    z = np.add.reduceat(e, D.start)
    return e / z[seg], u, m + np.log(z), seg

def loglik(D: Design, beta: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones(len(D.start)) if w is None else w
    _, u, lse, _ = _probs(D, beta)
    return float(np.sum(w * (u[D.chosen] - lse)))

def null_loglik(D: Design, w: np.ndarray | None = None) -> float:
    w = np.ones(len(D.start)) if w is None else w
    return float(-np.sum(w * np.log(D.sizes)))

def fit(D: Design, w: np.ndarray | None = None, beta0: np.ndarray | None = None,
        ridge: float = 1e-6, tol: float = 1e-9, max_iter: int = 50) -> tuple[np.ndarray, float]:
    p = D.X.shape[1]
    w = np.ones(len(D.start)) if w is None else np.asarray(w, dtype=np.float64)
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=np.float64)
    if p == 0:
        return beta, loglik(D, beta, w)
    prev = -np.inf
    for _ in range(max_iter):
        P, u, lse, seg = _probs(D, beta)
        ll = float(np.sum(w * (u[D.chosen] - lse))) - 0.5 * ridge * beta @ beta
        wr = w[seg] * P
        xbar = np.add.reduceat(D.X * P[:, None], D.start)
        g = (w[:, None] * (D.X[D.chosen] - xbar)).sum(0) - ridge * beta
        Hn = (D.X * wr[:, None]).T @ D.X - (xbar * w[:, None]).T @ xbar + ridge * np.eye(p)
        step = np.linalg.solve(Hn, g)
        t = 1.0
        while True:
            nb = beta + t * step
            if loglik(D, nb, w) - 0.5 * ridge * nb @ nb >= ll - 1e-12 or t < 1e-4:
                break
            t /= 2
        beta = nb
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    return beta, loglik(D, beta, w)

def drop_one(D: Design, blocks: dict[str, list[str]], w: np.ndarray | None = None,
             beta_full: np.ndarray | None = None) -> tuple[np.ndarray, float, dict[str, float]]:
    beta, ll = fit(D, w, beta_full)
    gains = {}
    for name, cols in blocks.items():
        keep = [k for k in D.names if k not in cols]
        b0 = np.array([beta[D.names.index(k)] for k in keep])
        _, ll_k = fit(D.columns(keep), w, b0)
        gains[name] = ll - ll_k
    return beta, ll, gains

def cell_weights(cell: np.ndarray, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    mult = np.bincount(rng.integers(0, n_cells, n_cells), minlength=n_cells).astype(np.float64)
    return mult[cell]
