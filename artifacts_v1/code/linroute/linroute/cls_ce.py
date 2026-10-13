from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from linroute.backend import as_host, get_backend
from linroute.probes import (
    _class_macro_mean,
    _fit_basis,
    _nested_fold_plan,
    class_macro_accuracy,
    feasible_q,
)

__all__ = [
    "CEConfig",
    "FIXED_LAMBDA",
    "ce_objective_components",
    "ce_objective_gradient",
    "solve_ce",
    "solve_ce_batch",
    "scipy_reference_fit",
    "select_pair",
    "lin_cls_ce_nested",
]

SCHEMA = "linroute/lin-cls-ce/3"
OBJECTIVE = "mean_unweighted_nll_plus_lambda_over_2_frobenius_w2"
INTERCEPT_GAUGE = "b_equals_b_raw_minus_mean_b_raw"
FIXED_LAMBDA = 1e-2
SELECTION_DIMENSION = "layer_only"
TIE_RULE = "lower_layer"
CURVATURE_RULE = "sty_gt_max_epsilon_times_norm_product_and_float_tiny"
RESUME_SCHEMA = "linroute/lin-cls-ce-resume/3"

RIDGE_PREPROCESS_SOURCE_COMMIT = "f1d5ab770ce97e71d63e1f922829db92bdf55cd1"
RIDGE_PREPROCESS_SYMBOLS = (
    "linroute.probes._fit_basis",
    "linroute.probes.feasible_q",
    "linroute.probes._nested_fold_plan",
)

@dataclass(frozen=True)
class CEConfig:
    q: int = 16
    lambdas: tuple[float, ...] = (FIXED_LAMBDA,)
    outer_folds: int = 10
    inner_folds: int = 5
    folds_seed: int = 0
    layers: tuple[int, ...] | None = None
    whitening_epsilon: float = 1e-8
    whitening_ddof: int = 0
    grad_atol: float = 1e-8
    grad_rtol: float = 1e-8
    maxiter: int = 1000
    history_size: int = 10
    armijo: float = 1e-4
    line_search_shrink: float = 0.5
    line_search_max_steps: int = 40
    min_step: float = 2.0 ** -40
    curvature_epsilon: float = 1e-12
    backend: str = "torch"
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.q <= 0:
            raise ValueError("q must be positive")
        if not self.lambdas or any(not np.isfinite(x) or x <= 0 for x in self.lambdas):
            raise ValueError("lambdas must be finite and positive")
        if len(set(self.lambdas)) != len(self.lambdas):
            raise ValueError("lambdas must be distinct")
        if self.whitening_ddof != 0:
            raise ValueError("the frozen ridge adapter uses population std (ddof=0)")
        if not np.isfinite(self.whitening_epsilon) or self.whitening_epsilon <= 0:
            raise ValueError("whitening_epsilon must be finite and positive")
        if self.backend != "torch":
            raise ValueError("LIN-CLS-CE production solver requires backend='torch'")
        if self.dtype != "float64":
            raise ValueError("only the validated float64 route is implemented")
        if (not np.isfinite(self.grad_atol) or not np.isfinite(self.grad_rtol)
                or self.grad_atol < 0 or self.grad_rtol < 0):
            raise ValueError("gradient tolerances must be finite and non-negative")
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("outer_folds and inner_folds must both be at least two")
        if self.maxiter <= 0 or self.history_size <= 0:
            raise ValueError("maxiter and history_size must be positive")
        if not 0 < self.armijo < 1:
            raise ValueError("armijo must lie strictly between zero and one")
        if not 0 < self.line_search_shrink < 1:
            raise ValueError("line_search_shrink must lie strictly between zero and one")
        if (self.line_search_max_steps <= 0 or not np.isfinite(self.min_step)
                or self.min_step <= 0):
            raise ValueError("line-search limits must be finite and positive")
        if self.curvature_epsilon < 0 or not np.isfinite(self.curvature_epsilon):
            raise ValueError("curvature_epsilon must be finite and non-negative")
        if self.layers is not None and len(set(self.layers)) != len(self.layers):
            raise ValueError("candidate layers must be distinct")

    def as_dict(self) -> dict:
        out = asdict(self)
        out.update({
            "objective": OBJECTIVE,
            "intercept_gauge": INTERCEPT_GAUGE,
            "tie_rule": TIE_RULE,
            "selection_dimension": SELECTION_DIMENSION,
            "fixed_lambda": FIXED_LAMBDA,
            "zero_initialization": True,
            "residual": "projected_gradient_infinity_norm",
            "residual_gate": "r <= grad_atol + grad_rtol * r0",
            "curvature_rule": CURVATURE_RULE,
            "ridge_preprocess_source_commit": RIDGE_PREPROCESS_SOURCE_COMMIT,
            "ridge_preprocess_symbols": list(RIDGE_PREPROCESS_SYMBOLS),
        })
        out["lambdas"] = list(self.lambdas)
        out["layers"] = None if self.layers is None else list(self.layers)
        return out

def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"torch has no dtype {name!r}") from exc
    if dtype is not torch.float64:
        raise ValueError("LIN-CLS-CE is frozen to torch.float64")
    return dtype

def _curvature_accepted(
    sty: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    tiny = torch.full_like(scale, torch.finfo(scale.dtype).tiny)
    floor = torch.maximum(epsilon * scale, tiny)
    return (
        torch.isfinite(sty)
        & torch.isfinite(scale)
        & (scale >= 0)
        & (sty > floor)
    )

def _as_fit_batch(
    X, y, lambdas, *, device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.as_tensor(X, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    lam = torch.as_tensor(lambdas, dtype=dtype, device=device).reshape(-1)
    if X.ndim == 2:
        X = X.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if X.ndim != 3 or y.ndim != 2:
        raise ValueError("X must be (B,n,q) and y must be (B,n)")
    if X.shape[:2] != y.shape:
        raise ValueError(f"X/y shape mismatch: {tuple(X.shape)} vs {tuple(y.shape)}")
    if X.shape[0] != len(lam):
        raise ValueError("one lambda is required per fit")
    if not torch.isfinite(X).all() or not torch.isfinite(lam).all():
        raise ValueError("X and lambdas must be finite")
    if (lam <= 0).any():
        raise ValueError("every lambda must be positive")
    return X.contiguous(), y.contiguous(), lam.contiguous()

def _split_theta(theta: torch.Tensor, q: int, n_classes: int):
    W = theta[:, :q * n_classes].reshape(-1, q, n_classes)
    b_raw = theta[:, q * n_classes:].reshape(-1, n_classes)

    b = b_raw - b_raw.mean(dim=-1, keepdim=True)
    return W, b

def _loss_grad_batch(
    theta: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    lambdas: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    B, n, q = X.shape
    W, b = _split_theta(theta, q, n_classes)
    logits = torch.bmm(X, W) + b[:, None, :]
    log_norm = torch.logsumexp(logits, dim=-1)
    gold = logits.gather(-1, y[..., None]).squeeze(-1)
    nll = (log_norm - gold).mean(dim=-1)
    penalty = 0.5 * lambdas * W.square().sum(dim=(1, 2))
    objective = nll + penalty

    probs = torch.softmax(logits, dim=-1)
    residual = probs.clone()
    residual.scatter_add_(
        -1, y[..., None],
        -torch.ones((B, n, 1), dtype=X.dtype, device=X.device),
    )
    residual /= float(n)
    gW = torch.bmm(X.transpose(1, 2), residual) + lambdas[:, None, None] * W
    gb = residual.sum(dim=1)
    gb = gb - gb.mean(dim=-1, keepdim=True)
    grad = torch.cat((gW.reshape(B, -1), gb), dim=1)
    row_sum_error = (probs.sum(dim=-1) - 1.0).abs().amax(dim=-1)
    parts = {
        "nll": nll,
        "penalty": penalty,
        "row_sum_error": row_sum_error,
        "b_mean_abs": b.mean(dim=-1).abs(),
    }
    return objective, grad, parts

def _loss_only_batch(
    theta: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    lambdas: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    _B, n, q = X.shape
    W, b = _split_theta(theta, q, n_classes)
    logits = torch.bmm(X, W) + b[:, None, :]
    nll = (
        torch.logsumexp(logits, dim=-1)
        - logits.gather(-1, y[..., None]).squeeze(-1)
    ).mean(dim=-1)
    return nll + 0.5 * lambdas * W.square().sum(dim=(1, 2))

def ce_objective_components(X, y, W, b_raw, lam: float) -> dict[str, float]:
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.long)
    W = torch.as_tensor(W, dtype=torch.float64)
    b_raw = torch.as_tensor(b_raw, dtype=torch.float64)
    b = b_raw - b_raw.mean()
    logits = X @ W + b
    nll = (torch.logsumexp(logits, dim=-1) - logits[torch.arange(len(y)), y]).mean()
    penalty = 0.5 * float(lam) * W.square().sum()
    return {
        "nll": float(nll),
        "penalty": float(penalty),
        "objective": float(nll + penalty),
        "intercept_mean": float(b.mean()),
    }

def ce_objective_gradient(X, y, W, b_raw, lam: float) -> tuple[float, np.ndarray, np.ndarray]:
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.long)
    W = torch.as_tensor(W, dtype=torch.float64)
    b_raw = torch.as_tensor(b_raw, dtype=torch.float64)
    theta = torch.cat((W.reshape(1, -1), b_raw.reshape(1, -1)), dim=1)
    obj, grad, _ = _loss_grad_batch(
        theta, X.unsqueeze(0), y.unsqueeze(0),
        torch.tensor([float(lam)], dtype=torch.float64), W.shape[1],
    )
    cut = W.numel()
    return (
        float(obj[0]),
        grad[0, :cut].reshape(W.shape).cpu().numpy(),
        grad[0, cut:].cpu().numpy(),
    )

def _two_loop_direction(
    grad: torch.Tensor,
    history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    gamma: torch.Tensor,
) -> torch.Tensor:
    qv = grad.clone()
    alphas: list[torch.Tensor] = []
    for s, yv, rho, valid in reversed(history):
        alpha = rho * (s * qv).sum(dim=1)
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        qv = qv - alpha[:, None] * yv
        alphas.append(alpha)
    rv = gamma[:, None] * qv
    for (s, yv, rho, valid), alpha in zip(history, reversed(alphas)):
        beta = rho * (yv * rv).sum(dim=1)
        beta = torch.where(valid, beta, torch.zeros_like(beta))
        rv = rv + s * (alpha - beta)[:, None]
    return -rv

def solve_ce_batch(
    X,
    y,
    lambdas,
    n_classes: int,
    cfg: CEConfig = CEConfig(),
) -> dict:
    dtype = _torch_dtype(cfg.dtype)
    X, y, lambdas = _as_fit_batch(
        X, y, lambdas, device=cfg.device, dtype=dtype
    )
    B, n, q = X.shape
    if n_classes < 2:
        raise ValueError("n_classes must be at least two")
    if y.numel() and (int(y.min()) < 0 or int(y.max()) >= n_classes):
        raise ValueError(f"labels must lie in 0..{n_classes - 1}")

    p = q * n_classes + n_classes
    theta = torch.zeros((B, p), dtype=dtype, device=cfg.device)
    objective, grad, parts = _loss_grad_batch(theta, X, y, lambdas, n_classes)
    r0 = grad.abs().amax(dim=1)
    threshold = cfg.grad_atol + cfg.grad_rtol * r0

    finite = torch.isfinite(objective) & torch.isfinite(grad).all(dim=1)
    converged = finite & (r0 <= threshold)
    active = finite & ~converged
    status = [
        "converged_gradient" if bool(converged[i]) else
        ("running" if bool(active[i]) else "nonfinite_initial")
        for i in range(B)
    ]
    stop_reason = [s for s in status]
    iterations = torch.zeros(B, dtype=torch.long, device=cfg.device)
    function_evals = torch.ones(B, dtype=torch.long, device=cfg.device)
    gradient_evals = torch.ones(B, dtype=torch.long, device=cfg.device)
    line_search_failures = torch.zeros(B, dtype=torch.long, device=cfg.device)
    curvature_pairs_accepted = torch.zeros(B, dtype=torch.long, device=cfg.device)
    curvature_pairs_rejected = torch.zeros(B, dtype=torch.long, device=cfg.device)
    gamma = torch.ones(B, dtype=dtype, device=cfg.device)
    history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for _global_iteration in range(cfg.maxiter):
        if not bool(active.any()):
            break
        direction = _two_loop_direction(grad, history, gamma)
        gtd = (grad * direction).sum(dim=1)
        bad_direction = active & (~torch.isfinite(gtd) | (gtd >= 0))
        if bool(bad_direction.any()):
            direction[bad_direction] = -grad[bad_direction]
            gtd[bad_direction] = -(grad[bad_direction].square().sum(dim=1))

            history = [
                (s, yy, rho, valid & ~bad_direction)
                for s, yy, rho, valid in history
            ]
            gamma[bad_direction] = 1.0

        step = torch.ones(B, dtype=dtype, device=cfg.device)
        pending = active.clone()
        accepted = torch.zeros(B, dtype=torch.bool, device=cfg.device)
        trial_theta = theta.clone()
        trial_objective = objective.clone()

        for _ls in range(cfg.line_search_max_steps):
            idx = torch.nonzero(pending, as_tuple=False).flatten()
            if len(idx) == 0:
                break
            candidate = theta[idx] + step[idx, None] * direction[idx]
            cand_obj = _loss_only_batch(
                candidate, X[idx], y[idx], lambdas[idx], n_classes
            )
            function_evals[idx] += 1
            okay = torch.isfinite(cand_obj) & (
                cand_obj <= objective[idx] + cfg.armijo * step[idx] * gtd[idx]
            )
            if bool(okay.any()):
                good = idx[okay]
                trial_theta[good] = candidate[okay]
                trial_objective[good] = cand_obj[okay]
                accepted[good] = True
                pending[good] = False
            rejected = idx[~okay]
            if len(rejected):
                step[rejected] *= cfg.line_search_shrink
                too_small = step[rejected] < cfg.min_step
                if bool(too_small.any()):
                    pending[rejected[too_small]] = False

        failed = active & ~accepted
        if bool(failed.any()):
            line_search_failures[failed] += 1
            active[failed] = False
            for i in torch.nonzero(failed, as_tuple=False).flatten().tolist():
                status[i] = "line_search_failed"
                stop_reason[i] = "armijo_no_acceptable_step"

        if not bool(accepted.any()):
            continue
        idx = torch.nonzero(accepted, as_tuple=False).flatten()
        new_obj, new_grad, new_parts = _loss_grad_batch(
            trial_theta[idx], X[idx], y[idx], lambdas[idx], n_classes
        )
        gradient_evals[idx] += 1

        function_evals[idx] += 1
        new_finite = torch.isfinite(new_obj) & torch.isfinite(new_grad).all(dim=1)
        nonfinite = idx[~new_finite]
        if len(nonfinite):
            active[nonfinite] = False
            for i in nonfinite.tolist():
                status[i] = "nonfinite_iteration"
                stop_reason[i] = "nonfinite_objective_or_gradient"

        good = idx[new_finite]
        if len(good) == 0:
            continue
        local_good = torch.nonzero(new_finite, as_tuple=False).flatten()
        old_theta = theta[good].clone()
        old_grad = grad[good].clone()
        theta[good] = trial_theta[good]
        objective[good] = new_obj[local_good]
        grad[good] = new_grad[local_good]
        for key in parts:
            parts[key][good] = new_parts[key][local_good]
        iterations[good] += 1

        s_good = theta[good] - old_theta
        y_good = grad[good] - old_grad
        sty = (s_good * y_good).sum(dim=1)
        scale = s_good.norm(dim=1) * y_good.norm(dim=1)
        curvature_ok = _curvature_accepted(
            sty, scale, cfg.curvature_epsilon
        )
        curvature_pairs_accepted[good] += curvature_ok.to(torch.long)
        curvature_pairs_rejected[good] += (~curvature_ok).to(torch.long)
        s_full = torch.zeros_like(theta)
        y_full = torch.zeros_like(theta)
        rho_full = torch.zeros(B, dtype=dtype, device=cfg.device)
        valid_full = torch.zeros(B, dtype=torch.bool, device=cfg.device)
        curved = good[curvature_ok]
        if len(curved):
            loc = torch.nonzero(curvature_ok, as_tuple=False).flatten()
            s_full[curved] = s_good[loc]
            y_full[curved] = y_good[loc]
            rho_full[curved] = 1.0 / sty[loc]
            valid_full[curved] = True
            yy = y_good[loc].square().sum(dim=1)
            gamma[curved] = torch.where(yy > 0, sty[loc] / yy, torch.ones_like(yy))
        history.append((s_full, y_full, rho_full, valid_full))
        if len(history) > cfg.history_size:
            history.pop(0)

        r = grad[good].abs().amax(dim=1)
        done_local = r <= threshold[good]
        done = good[done_local]
        if len(done):
            active[done] = False
            converged[done] = True
            for i in done.tolist():
                status[i] = "converged_gradient"
                stop_reason[i] = "projected_gradient_gate"

    exhausted = active.clone()
    if bool(exhausted.any()):
        active[exhausted] = False
        for i in torch.nonzero(exhausted, as_tuple=False).flatten().tolist():
            status[i] = "maxiter"
            stop_reason[i] = "maximum_iterations_without_gradient_gate"

    final_objective, final_grad, final_parts = _loss_grad_batch(
        theta, X, y, lambdas, n_classes
    )
    function_evals += 1
    gradient_evals += 1
    residual = final_grad.abs().amax(dim=1)
    final_finite = (
        torch.isfinite(final_objective)
        & torch.isfinite(final_grad).all(dim=1)
        & torch.isfinite(theta).all(dim=1)
    )
    converged = final_finite & (residual <= threshold) & torch.as_tensor(
        [s == "converged_gradient" for s in status], device=cfg.device
    )
    W, b = _split_theta(theta, q, n_classes)
    logits = torch.bmm(X, W) + b[:, None, :]
    pred = logits.argmax(dim=-1)

    diagnostics = []
    for i in range(B):
        diagnostics.append({
            "status": status[i],
            "converged": bool(converged[i]),
            "stop_reason": stop_reason[i],
            "objective": float(final_objective[i]),
            "nll": float(final_parts["nll"][i]),
            "penalty": float(final_parts["penalty"][i]),
            "projected_grad_inf": float(residual[i]),
            "projected_grad_inf_initial": float(r0[i]),
            "projected_grad_threshold": float(threshold[i]),
            "iterations": int(iterations[i]),
            "function_evals": int(function_evals[i]),
            "gradient_evals": int(gradient_evals[i]),
            "line_search_failures": int(line_search_failures[i]),
            "curvature_pairs_accepted": int(curvature_pairs_accepted[i]),
            "curvature_pairs_rejected": int(curvature_pairs_rejected[i]),
            "max_probability_rowsum_error": float(final_parts["row_sum_error"][i]),
            "intercept_mean_abs": float(final_parts["b_mean_abs"][i]),
            "finite_objective": bool(torch.isfinite(final_objective[i])),
            "finite_gradient": bool(torch.isfinite(final_grad[i]).all()),
            "finite_parameters": bool(torch.isfinite(theta[i]).all()),
            "lambda": float(lambdas[i]),
            "n": int(n),
            "q": int(q),
            "n_classes": int(n_classes),
        })
    return {
        "W": W.detach().cpu().numpy(),
        "b": b.detach().cpu().numpy(),
        "pred_train": pred.detach().cpu().numpy(),
        "diagnostics": diagnostics,
    }

def solve_ce(X, y, n_classes: int, lam: float, cfg: CEConfig = CEConfig()) -> dict:
    out = solve_ce_batch(X, y, [lam], n_classes, cfg)
    return {
        "W": out["W"][0],
        "b": out["b"][0],
        "pred_train": out["pred_train"][0],
        "diagnostics": out["diagnostics"][0],
    }

def scipy_reference_fit(
    X,
    y,
    n_classes: int,
    lam: float,
    cfg: CEConfig = CEConfig(),
) -> dict:
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, q = X.shape
    p = q * n_classes + n_classes

    def fun(theta: np.ndarray):
        W = theta[:q * n_classes].reshape(q, n_classes)
        b_raw = theta[q * n_classes:]
        b = b_raw - b_raw.mean()
        logits = X @ W + b
        shift = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - shift)
        probs = exp / exp.sum(axis=1, keepdims=True)
        log_norm = np.log(exp.sum(axis=1)) + shift[:, 0]
        nll = np.mean(log_norm - logits[np.arange(n), y])
        objective = nll + 0.5 * float(lam) * np.square(W).sum()
        residual = probs.copy()
        residual[np.arange(n), y] -= 1.0
        residual /= float(n)
        gW = X.T @ residual + float(lam) * W
        gb = residual.sum(axis=0)
        gb -= gb.mean()
        return float(objective), np.concatenate((gW.ravel(), gb))

    zero = np.zeros(p, dtype=np.float64)
    _, g0 = fun(zero)
    r0 = float(np.max(np.abs(g0)))
    threshold = cfg.grad_atol + cfg.grad_rtol * r0
    result = minimize(
        fun, zero, method="L-BFGS-B", jac=True,
        options={"maxiter": cfg.maxiter, "gtol": cfg.grad_atol, "ftol": 0.0,
                 "maxls": cfg.line_search_max_steps},
    )
    objective, grad = fun(result.x)
    residual = float(np.max(np.abs(grad)))
    W = result.x[:q * n_classes].reshape(q, n_classes)
    b_raw = result.x[q * n_classes:]
    b = b_raw - b_raw.mean()
    logits = X @ W + b
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    finite = bool(np.isfinite(result.x).all() and np.isfinite(objective)
                  and np.isfinite(grad).all())
    converged = bool(finite and residual <= threshold)
    return {
        "W": W,
        "b": b,
        "pred_train": logits.argmax(axis=1),
        "prob_train": probs,
        "diagnostics": {
            "status": "converged_gradient" if converged else "nonconverged",
            "converged": converged,
            "stop_reason": str(result.message),
            "objective": objective,
            "projected_grad_inf": residual,
            "projected_grad_inf_initial": r0,
            "projected_grad_threshold": threshold,
            "iterations": int(result.nit),
            "function_evals": int(result.nfev),
            "gradient_evals": int(result.njev),
            "line_search_failures": None,
            "max_probability_rowsum_error": float(
                np.max(np.abs(probs.sum(axis=1) - 1.0))
            ),
            "intercept_mean_abs": float(abs(b.mean())),
            "finite_objective": bool(np.isfinite(objective)),
            "finite_gradient": bool(np.isfinite(grad).all()),
            "finite_parameters": bool(np.isfinite(result.x).all()),
            "library_success": bool(result.success),
            "library_status": int(result.status),
        },
    }

def select_pair(scores: dict[tuple[int, float], float]) -> tuple[int, float]:
    candidates = [
        (layer, lam, float(score))
        for (layer, lam), score in scores.items() if np.isfinite(score)
    ]
    if not candidates:
        raise ValueError("no finite layer score is available")
    if any(lam != FIXED_LAMBDA for _layer, lam, _score in candidates):
        raise ValueError(f"LIN-CLS-CE /3 fixes lambda at {FIXED_LAMBDA}")
    layer, lam, _ = min(candidates, key=lambda x: (-x[2], x[0]))
    return int(layer), float(lam)

def _complement(n: int, heldout: np.ndarray) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(heldout, dtype=np.int64)] = False
    return np.flatnonzero(mask)

def _missing_classes(labels: np.ndarray, rows: np.ndarray, n_classes: int) -> list[int]:
    present = set(np.asarray(labels)[np.asarray(rows, dtype=np.int64)].tolist())
    return [c for c in range(n_classes) if c not in present]

def _project_layers(
    S: torch.Tensor,
    fitting_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    layers: Sequence[int],
    cfg: CEConfig,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, dict]]:
    B = get_backend("torch", cfg.device, cfg.dtype)
    fitting_rows = np.asarray(fitting_rows, dtype=np.int64)
    evaluation_rows = np.asarray(evaluation_rows, dtype=np.int64)
    layer_idx = np.asarray(list(layers), dtype=np.int64)

    X_fit = B.contiguous(
        B.swapaxes(B.take(B.take(S, fitting_rows, 0), layer_idx, 1), 0, 1)
    )
    X_eval = B.contiguous(
        B.swapaxes(B.take(B.take(S, evaluation_rows, 0), layer_idx, 1), 0, 1)
    )
    mu, Vt, singular = _fit_basis(X_fit, backend=B)
    singular_host = as_host(singular)
    projected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    provenance: dict[int, dict] = {}
    for li, layer in enumerate(layers):
        s = singular_host[li]
        rank = 0
        if s.size and s[0] > 0:
            tol = s[0] * max(len(fitting_rows), S.shape[-1]) * np.finfo(np.float64).eps
            rank = int((s > tol).sum())
        rank_feasible = feasible_q(s, len(fitting_rows), int(S.shape[-1]), cfg.q)
        record = {
            "rank_feasible": bool(rank_feasible),
            "projection_finite": None,
            "feasible": bool(rank_feasible),
            "n_fitting": int(len(fitting_rows)),
            "n_evaluation": int(len(evaluation_rows)),
            "centered_numeric_rank": rank,
            "q": int(cfg.q),
        }
        if rank_feasible:
            V = Vt[li, :cfg.q, :].transpose(-1, -2)
            Z_fit = (X_fit[li] - mu[li]) @ V
            sd = Z_fit.std(dim=0, correction=cfg.whitening_ddof) + cfg.whitening_epsilon
            Z_fit = Z_fit / sd
            Z_eval = ((X_eval[li] - mu[li]) @ V) / sd
            projected[int(layer)] = (Z_fit.contiguous(), Z_eval.contiguous())
            record.update({
                "whitening_sd_min": float(sd.min()),
                "whitening_sd_max": float(sd.max()),
                "finite": bool(torch.isfinite(Z_fit).all() and torch.isfinite(Z_eval).all()),
            })
            record["projection_finite"] = record["finite"]
            if not record["finite"]:
                projected.pop(int(layer))
                record["feasible"] = False
                record["numerical_failure"] = "nonfinite_projected_features"
        provenance[int(layer)] = record
    return projected, provenance

def _predict(X: torch.Tensor, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    Wt = torch.as_tensor(W, dtype=X.dtype, device=X.device)
    bt = torch.as_tensor(b, dtype=X.dtype, device=X.device)
    return (X @ Wt + bt).argmax(dim=-1).cpu().numpy()

def _diag_batch(
    phase: str,
    outer: int,
    inner: int | None,
    label_set: int,
    combos: Sequence[tuple[int, float]],
    diagnostics: Sequence[dict],
) -> dict:
    return {
        "phase": phase,
        "outer": int(outer),
        "inner": None if inner is None else int(inner),
        "label_set": int(label_set),
        "fits": [
            {"layer": int(layer), "lambda": float(lam), **diag}
            for (layer, lam), diag in zip(combos, diagnostics)
        ],
    }

def lin_cls_ce_nested(
    S_all,
    label_sets: Sequence[np.ndarray],
    n_classes: int,
    cfg: CEConfig = CEConfig(),
    *,
    folds: Sequence[Sequence[int]] | None = None,
    inner_folds: Sequence[Sequence[Sequence[int]]] | None = None,
    resume_state: dict | None = None,
    progress_callback=None,
    scipy_oracle: bool = False,
) -> dict:
    if tuple(cfg.lambdas) != (FIXED_LAMBDA,):
        raise ValueError(
            f"LIN-CLS-CE /3 requires fixed lambdas=({FIXED_LAMBDA},); "
            "the retired /2 joint grid cannot be reused"
        )
    dtype = _torch_dtype(cfg.dtype)
    S = torch.as_tensor(S_all, dtype=dtype, device=cfg.device).contiguous()
    if S.ndim != 3:
        raise ValueError("S_all must have shape (n, layer, d)")
    n, n_layer_axis, _d = S.shape
    labels = [np.asarray(x, dtype=np.int64) for x in label_sets]
    if not labels:
        raise ValueError("at least one label set is required")
    for lab in labels:
        if lab.shape != (n,):
            raise ValueError("every label set must match the subject axis")
        if lab.size and (lab.min() < 0 or lab.max() >= n_classes):
            raise ValueError(f"labels must lie in 0..{n_classes - 1}")
    layers = tuple(cfg.layers) if cfg.layers is not None else tuple(range(n_layer_axis))
    if not layers or any(layer < 0 or layer >= n_layer_axis for layer in layers):
        raise ValueError("candidate layers must lie on the state layer axis")
    if len(set(layers)) != len(layers):
        raise ValueError("candidate layers must be distinct")

    class _FoldConfig:
        outer_folds = cfg.outer_folds
        inner_folds = cfg.inner_folds
        folds_seed = cfg.folds_seed

    plan = _nested_fold_plan(n, _FoldConfig(), folds, inner_folds)
    n_sets = len(labels)
    if resume_state is None:
        pred = [np.full(n, -1, dtype=np.int64) for _ in labels]
        majority = [np.zeros(n, dtype=bool) for _ in labels]
        selected: list[list[dict]] = [[] for _ in labels]
        gates: list[list[dict]] = [[] for _ in labels]
        fit_diagnostics: list[dict] = []
        projection_diagnostics: list[dict] = []
        oracle_checks: list[dict] = []
        unavailable = [False] * n_sets
        completed_outer: set[int] = set()
        candidate_grid_counts = {
            "expected": 0,
            "attempted": 0,
            "converged": 0,
            "failed": 0,
            "projection_failed": 0,
            "solver_failed": 0,
            "by_label_set": [
                {
                    "expected": 0,
                    "attempted": 0,
                    "converged": 0,
                    "failed": 0,
                    "projection_failed": 0,
                    "solver_failed": 0,
                }
                for _ in labels
            ],
        }
    else:
        if resume_state.get("schema") != RESUME_SCHEMA:
            raise ValueError("resume state is not LIN-CLS-CE /3")
        if resume_state.get("n") != n or resume_state.get("n_label_sets") != n_sets:
            raise ValueError("resume state does not match the requested subject/label axes")
        if bool(resume_state.get("scipy_oracle", False)) != bool(scipy_oracle):
            raise ValueError("resume state does not match the requested oracle mode")
        pred = [np.asarray(x, dtype=np.int64) for x in resume_state["pred"]]
        majority = [np.asarray(x, dtype=bool) for x in resume_state["majority"]]
        selected = [list(x) for x in resume_state["selected"]]
        gates = [list(x) for x in resume_state["gates"]]
        fit_diagnostics = list(resume_state["fit_diagnostics"])
        projection_diagnostics = list(resume_state["projection_diagnostics"])
        oracle_checks = list(resume_state.get("oracle_checks", []))
        unavailable = [bool(x) for x in resume_state["unavailable"]]
        completed_outer = {int(x) for x in resume_state["completed_outer"]}
        candidate_grid_counts = {
            "expected": int(resume_state["candidate_grid_counts"]["expected"]),
            "attempted": int(resume_state["candidate_grid_counts"]["attempted"]),
            "converged": int(resume_state["candidate_grid_counts"]["converged"]),
            "failed": int(resume_state["candidate_grid_counts"]["failed"]),
            "projection_failed": int(
                resume_state["candidate_grid_counts"]["projection_failed"]
            ),
            "solver_failed": int(
                resume_state["candidate_grid_counts"]["solver_failed"]
            ),
            "by_label_set": [
                {key: int(value) for key, value in counts.items()}
                for counts in resume_state["candidate_grid_counts"]["by_label_set"]
            ],
        }

    def checkpoint_state() -> dict:
        return {
            "schema": RESUME_SCHEMA,
            "n": int(n),
            "n_label_sets": int(n_sets),
            "scipy_oracle": bool(scipy_oracle),
            "completed_outer": sorted(completed_outer),
            "pred": [x.tolist() for x in pred],
            "majority": [x.tolist() for x in majority],
            "selected": selected,
            "gates": gates,
            "fit_diagnostics": fit_diagnostics,
            "projection_diagnostics": projection_diagnostics,
            "oracle_checks": oracle_checks,
            "unavailable": unavailable,
            "candidate_grid_counts": candidate_grid_counts,
        }

    for outer_i, (te, tr, inner) in enumerate(plan):
        if outer_i in completed_outer:
            continue
        projected_outer, outer_proj = _project_layers(S, tr, te, layers, cfg)
        projection_diagnostics.append({
            "outer": outer_i, "inner": None,
            "layers": {str(k): v for k, v in outer_proj.items()},
        })

        inner_data = []
        structural_layers = {
            int(layer) for layer, record in outer_proj.items()
            if record["rank_feasible"]
        }
        for inner_i, te_local in enumerate(inner):
            tr_local = _complement(len(tr), te_local)
            global_fit = tr[tr_local]
            global_eval = tr[te_local]
            projected, proj = _project_layers(
                S, global_fit, global_eval, layers, cfg
            )
            projection_diagnostics.append({
                "outer": outer_i, "inner": inner_i,
                "layers": {str(k): v for k, v in proj.items()},
            })
            structural_layers.intersection_update(
                int(layer) for layer, record in proj.items()
                if record["rank_feasible"]
            )
            inner_data.append((inner_i, te_local, global_fit, projected, proj))

        combos = [
            (layer, lam) for layer in layers if layer in structural_layers
            for lam in cfg.lambdas
        ]
        structural_pairs = set(combos)
        outer_projection_failed_layers = sorted(
            layer for layer in structural_layers if layer not in projected_outer
        )

        for j, lab in enumerate(labels):
            missing_gates = []
            missing = _missing_classes(lab, tr, n_classes)
            if missing:
                missing_gates.append({
                    "outer": outer_i, "inner": None,
                    "status": "missing_declared_classes",
                    "missing_classes": missing,
                })
            for inner_i, _te_local, global_fit, _projected, _proj in inner_data:
                missing = _missing_classes(lab, global_fit, n_classes)
                if missing:
                    missing_gates.append({
                        "outer": outer_i, "inner": inner_i,
                        "status": "missing_declared_classes",
                        "missing_classes": missing,
                    })
            if missing_gates:
                unavailable[j] = True
                gates[j].extend(missing_gates)
                selected[j].append({
                    "outer": outer_i,
                    "available": False,
                    "reason": "missing_declared_classes",
                    "inner_scores": [],
                })
                continue
            if not combos:
                unavailable[j] = True
                selected[j].append({
                    "outer": outer_i,
                    "available": False,
                    "reason": "no_structurally_rank_feasible_candidate",
                    "inner_scores": [
                        {
                            "layer": int(layer),
                            "lambda": float(lam),
                            "structurally_feasible": False,
                            "available": False,
                            "inner_macro_acc": None,
                        }
                        for layer in layers for lam in cfg.lambdas
                    ],
                })
                continue

            inner_pred = {
                pair: np.full(len(tr), -1, dtype=np.int64) for pair in combos
            }
            pair_converged = {pair: True for pair in combos}
            candidate_projection_failed = 0
            expected = len(combos) * len(inner_data)
            candidate_grid_counts["expected"] += expected
            candidate_grid_counts["by_label_set"][j]["expected"] += expected

            for inner_i, te_local, global_fit, projected, _proj in inner_data:
                available_combos = [
                    pair for pair in combos if pair[0] in projected
                ]
                projection_failed = len(combos) - len(available_combos)
                if projection_failed:
                    candidate_projection_failed += projection_failed
                    for pair in combos:
                        if pair[0] not in projected:
                            pair_converged[pair] = False
                    for key, value in (
                        ("failed", projection_failed),
                        ("projection_failed", projection_failed),
                    ):
                        candidate_grid_counts[key] += value
                        candidate_grid_counts["by_label_set"][j][key] += value
                if not available_combos:
                    continue
                X_batch = torch.stack([
                    projected[layer][0] for layer, _ in available_combos
                ])
                y_batch = np.stack([lab[global_fit] for _ in available_combos])
                solved = solve_ce_batch(
                    X_batch,
                    y_batch,
                    [lam for _, lam in available_combos],
                    n_classes,
                    cfg,
                )
                fit_diagnostics.append(_diag_batch(
                    "inner", outer_i, inner_i, j, available_combos,
                    solved["diagnostics"]
                ))
                attempted = len(available_combos)
                converged = sum(
                    bool(diag["converged"]) for diag in solved["diagnostics"]
                )
                solver_failed = attempted - converged
                for key, value in (
                    ("attempted", attempted),
                    ("converged", converged),
                    ("failed", solver_failed),
                    ("solver_failed", solver_failed),
                ):
                    candidate_grid_counts[key] += value
                    candidate_grid_counts["by_label_set"][j][key] += value
                for k, pair in enumerate(available_combos):
                    diag = solved["diagnostics"][k]
                    if not diag["converged"]:
                        pair_converged[pair] = False
                        continue
                    layer, _lam = pair
                    inner_pred[pair][te_local] = _predict(
                        projected[layer][1], solved["W"][k], solved["b"][k]
                    )

            lab_tr = lab[tr]
            scores: dict[tuple[int, float], float] = {}
            raw_scores = []
            for layer in layers:
                for lam in cfg.lambdas:
                    pair = (layer, lam)
                    structural = pair in structural_pairs
                    p = inner_pred.get(pair)
                    ok = bool(
                        structural
                        and pair_converged[pair]
                        and p is not None
                        and (p >= 0).all()
                    )
                    score = class_macro_accuracy(p, lab_tr) if ok else float("nan")
                    if ok:
                        scores[pair] = score
                    raw_scores.append({
                        "layer": int(layer), "lambda": float(lam),
                        "structurally_feasible": structural,
                        "available": ok,
                        "inner_macro_acc": None if not np.isfinite(score) else float(score),
                    })
            if not all(pair_converged.values()):
                unavailable[j] = True
                selected[j].append({
                    "outer": outer_i, "available": False,
                    "reason": (
                        "candidate_grid_projection_nonfinite"
                        if candidate_projection_failed
                        else "candidate_grid_nonconverged"
                    ),
                    "projection_failed": int(candidate_projection_failed),
                    "inner_scores": raw_scores,
                })
                continue
            if outer_projection_failed_layers:
                unavailable[j] = True
                selected[j].append({
                    "outer": outer_i,
                    "available": False,
                    "reason": "outer_projection_nonfinite",
                    "layers": outer_projection_failed_layers,
                    "inner_scores": raw_scores,
                })
                continue
            if not scores:
                unavailable[j] = True
                selected[j].append({
                    "outer": outer_i, "available": False,
                    "reason": "candidate_grid_incomplete",
                    "inner_scores": raw_scores,
                })
                continue
            best_layer, best_lam = select_pair(scores)
            X_fit, X_eval = projected_outer[best_layer]
            solved = solve_ce(
                X_fit, lab_tr, n_classes, best_lam, cfg
            )
            fit_diagnostics.append(_diag_batch(
                "outer_refit", outer_i, None, j, [(best_layer, best_lam)],
                [solved["diagnostics"]],
            ))
            if scipy_oracle:
                reference = scipy_reference_fit(
                    X_fit.detach().cpu().numpy(), lab_tr, n_classes,
                    best_lam, cfg,
                )
                production_logits = (
                    X_fit.detach().cpu().numpy() @ solved["W"] + solved["b"]
                )
                production_prob = np.exp(
                    production_logits
                    - production_logits.max(axis=1, keepdims=True)
                )
                production_prob /= production_prob.sum(axis=1, keepdims=True)
                objective_abs = abs(
                    solved["diagnostics"]["objective"]
                    - reference["diagnostics"]["objective"]
                )
                probability_max_abs = float(np.max(np.abs(
                    production_prob - reference["prob_train"]
                )))
                prediction_match = bool(np.array_equal(
                    solved["pred_train"], reference["pred_train"]
                ))
                objective_gate = 1e-6 * (
                    1.0 + abs(reference["diagnostics"]["objective"])
                )
                passed = bool(
                    solved["diagnostics"]["converged"]
                    and reference["diagnostics"]["converged"]
                    and objective_abs <= objective_gate
                    and probability_max_abs <= 1e-6
                    and prediction_match
                )
                oracle_checks.append({
                    "outer": int(outer_i),
                    "label_set": int(j),
                    "layer": int(best_layer),
                    "lambda": float(best_lam),
                    "passed": passed,
                    "objective_abs_difference": float(objective_abs),
                    "objective_gate": float(objective_gate),
                    "probability_max_abs_difference": probability_max_abs,
                    "probability_gate": 1e-6,
                    "training_prediction_match": prediction_match,
                    "production": solved["diagnostics"],
                    "scipy": reference["diagnostics"],
                })
            if not solved["diagnostics"]["converged"]:
                unavailable[j] = True
                selected[j].append({
                    "outer": outer_i, "available": False,
                    "reason": "outer_refit_nonconverged",
                    "layer": best_layer, "lambda": best_lam,
                    "inner_macro_acc": scores[(best_layer, best_lam)],
                    "inner_scores": raw_scores,
                })
                continue
            pred[j][te] = _predict(X_eval, solved["W"], solved["b"])
            counts = np.bincount(lab_tr, minlength=n_classes)
            majority_class = int(counts.argmax())
            majority[j][te] = lab[te] == majority_class
            selected[j].append({
                "outer": outer_i, "available": True,
                "layer": best_layer, "lambda": best_lam,
                "inner_macro_acc": scores[(best_layer, best_lam)],
                "inner_scores": raw_scores,
            })

        completed_outer.add(outer_i)
        if progress_callback is not None:
            progress_callback(checkpoint_state())

    results = []
    for j, lab in enumerate(labels):
        counts = np.bincount(lab, minlength=n_classes)
        available = bool(not unavailable[j] and (pred[j] >= 0).all())
        if available:
            acc = class_macro_accuracy(pred[j], lab)
            maj = _class_macro_mean(majority[j], lab)
            result = {
                "available": True,
                "acc": acc,
                "macro_acc": acc,
                "micro_acc": float((pred[j] == lab).mean()),
                "majority_acc": maj,
                "micro_majority_acc": float(majority[j].mean()),
                "gain": acc - maj,
                "micro_gain": float((pred[j] == lab).mean() - majority[j].mean()),
            }
        else:
            result = {
                "available": False,
                "reason": "one or more fitting partitions failed a locked gate",
                "acc": None, "macro_acc": None, "micro_acc": None,
                "majority_acc": None, "micro_majority_acc": None,
                "gain": None, "micro_gain": None,
            }
        result.update({
            "pred": pred[j],
            "selected_pairs": selected[j],
            "class_presence_gates": gates[j],
            "chance": 1.0 / float(n_classes),
            "majority_share": float(counts.max() / n),
            "n": int(n),
            "q": int(cfg.q),
            "n_classes": int(n_classes),
            "n_classes_present": int((counts > 0).sum()),
        })
        results.append(result)

    candidate_grid = {
        **candidate_grid_counts,
        "scope": (
            "inner fits at fixed lambda=1e-2 for every layer structurally "
            "rank-feasible in the outer refit and all inner fitting partitions"
        ),
        "complete": bool(
            candidate_grid_counts["attempted"]
            == candidate_grid_counts["expected"]
            and candidate_grid_counts["converged"]
            == candidate_grid_counts["expected"]
            and candidate_grid_counts["failed"] == 0
        ),
    }
    return {
        "schema": SCHEMA,
        "config": cfg.as_dict(),
        "results": results,
        "candidate_grid": candidate_grid,
        "fit_diagnostics": fit_diagnostics,
        "projection_diagnostics": projection_diagnostics,
        "oracle_checks": oracle_checks,
        "resume_state": checkpoint_state(),
    }
