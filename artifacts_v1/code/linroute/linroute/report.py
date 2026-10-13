from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "rankdata",
    "pearson",
    "spearman",
    "ols",
    "regression_rows",
    "write_csv",
    "write_org",
    "write_markdown",
]

CONFOUNDS = (
    ("entropy", lambda v: v, "entropy"),
    ("n_objects", np.log, "log_n_objects"),
    ("n", np.log, "log_n"),
)

def rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or sorted_x[i] != sorted_x[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / den) if den > 0 else float("nan")

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))

def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    X = np.asarray(X, dtype=np.float64).reshape(len(y), -1)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    design = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    dof = n - p - 1
    adj = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    return beta, r2, adj

def _design(rows: Sequence[dict]) -> np.ndarray:
    return np.column_stack(
        [np.array([fn(float(r[key])) for r in rows]) for key, fn, _ in CONFOUNDS]
    )

def _usable(rows: Sequence[dict], score: str, target: str) -> list[dict]:
    keys = [score, target] + [k for k, _, _ in CONFOUNDS]
    out = []
    for row in rows:
        vals = [row.get(k) for k in keys]
        if any(v is None for v in vals):
            continue
        vals = [float(v) for v in vals]
        if not all(np.isfinite(v) for v in vals):
            continue
        if row["n_objects"] <= 0 or row["n"] <= 0:
            continue
        out.append(row)
    return out

def regression_rows(
    rows: Sequence[dict],
    score_keys: Sequence[str],
    target_keys: Sequence[str],
) -> list[dict]:
    out: list[dict] = []
    for score in score_keys:
        for target in target_keys:
            use = _usable(rows, score, target)
            record = {"score": score, "target": target, "n": len(use)}

            if len(use) < 8:
                out.append(record)
                continue
            y = np.array([float(r[target]) for r in use])
            x = np.array([float(r[score]) for r in use])
            confounds = _design(use)
            _, _, adj_base = ols(confounds, y)
            beta, _, adj_aug = ols(np.column_stack([confounds, x]), y)
            sd = float(x.std())
            record.update({
                "coef": float(beta[-1]),
                "coef_per_sd": float(beta[-1] * sd),
                "adj_r2_base": adj_base,
                "adj_r2_aug": adj_aug,
                "adj_r2_gain": adj_aug - adj_base,
                "pearson": pearson(x, y),
                "spearman": spearman(x, y),
            })
            out.append(record)
    return out

def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "" if not np.isfinite(value) else f"{value:.{digits}f}"
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return str(value)

def write_csv(rows: Iterable[dict], columns: Sequence[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _fmt(row.get(c)) for c in columns})
    return path

def _delimited(rows: Iterable[dict], columns: Sequence[str], sep_row: str) -> str:
    body = ["| " + " | ".join(columns) + " |", sep_row]
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")
    return "\n".join(body) + "\n"

def write_org(rows: Iterable[dict], columns: Sequence[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "|" + "|".join("-" * (len(c) + 2) for c in columns) + "|"
    path.write_text(_delimited(list(rows), columns, sep), encoding="utf-8")
    return path

def write_markdown(rows: Iterable[dict], columns: Sequence[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    path.write_text(_delimited(list(rows), columns, sep), encoding="utf-8")
    return path
