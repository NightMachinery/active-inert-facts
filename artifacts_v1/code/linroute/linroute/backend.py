from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

__all__ = [
    "Backend",
    "LinAlgError",
    "NUMPY",
    "NumpyBackend",
    "TorchBackend",
    "as_host",
    "get_backend",
]

LinAlgError = np.linalg.LinAlgError

def as_host(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)

class Backend:
    name: str = "abstract"
    device: str = "cpu"
    dtype: str = "float64"

    @property
    def eps(self) -> float:
        return float(np.finfo(np.dtype(self.dtype)).eps)

    def describe(self) -> dict[str, str]:
        return {
            "compute_backend": self.name,
            "compute_device": self.device,
            "compute_dtype": self.dtype,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={self.device!r}, dtype={self.dtype!r})"

def _loop(fn, *arrays, batch: int):
    out = [fn(*[a[i] if getattr(a, "ndim", 0) > 2 else a for a in arrays])
           for i in range(batch)]
    if isinstance(out[0], tuple):
        return tuple(np.stack(parts) for parts in zip(*out))
    return np.stack(out)

@dataclass(frozen=True)
class NumpyBackend(Backend):
    name: str = "numpy"
    device: str = "cpu"
    dtype: str = "float64"

    def asarray(self, x) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach().to("cpu").numpy()
        return np.ascontiguousarray(np.asarray(x), dtype=np.dtype(self.dtype))

    def to_numpy(self, x) -> np.ndarray:
        return as_host(x)

    def eye(self, n: int) -> np.ndarray:
        return np.eye(n, dtype=np.dtype(self.dtype))

    def stack(self, arrays: Sequence) -> np.ndarray:
        return np.stack(list(arrays))

    def take(self, x, indices, axis: int):
        return np.take(x, np.asarray(indices, dtype=np.int64), axis=axis)

    def swapaxes(self, x, a: int, b: int):
        return np.swapaxes(x, a, b)

    def transpose(self, x):
        return np.swapaxes(x, -1, -2)

    def contiguous(self, x):
        return np.ascontiguousarray(x)

    def broadcast_to(self, x, shape):
        return np.broadcast_to(x, shape)

    def mean(self, x, axis: int, keepdims: bool = False):
        if x.ndim > 2:
            return _loop(lambda a: a.mean(axis, keepdims=keepdims), x,
                         batch=x.shape[0])
        return x.mean(axis, keepdims=keepdims)

    def sum(self, x, axis: int, keepdims: bool = False):
        if x.ndim > 2:
            return _loop(lambda a: a.sum(axis, keepdims=keepdims), x,
                         batch=x.shape[0])
        return x.sum(axis, keepdims=keepdims)

    def std(self, x, axis: int, keepdims: bool = False):
        if x.ndim > 2:
            return _loop(lambda a: a.std(axis, keepdims=keepdims), x,
                         batch=x.shape[0])
        return x.std(axis, keepdims=keepdims)

    def norm(self, x, axis: int):
        if x.ndim > 2:
            return _loop(lambda a: np.linalg.norm(a, axis=axis), x,
                         batch=x.shape[0])
        return np.linalg.norm(x, axis=axis)

    def sqrt(self, x):
        return np.sqrt(x)

    def clip(self, x, lo, hi):
        return np.clip(x, lo, hi)

    def where(self, cond, x, y):
        return np.where(cond, x, y)

    def matmul(self, a, b):
        batch = max(getattr(a, "ndim", 0), getattr(b, "ndim", 0))
        if batch > 2:
            n = a.shape[0] if a.ndim > 2 else b.shape[0]
            return _loop(lambda x, y: x @ y, a, b, batch=n)
        return a @ b

    def eigh(self, a):
        def one(m):
            w, W = np.linalg.eigh(m)
            order = np.argsort(w)[::-1]
            return w[order], W[:, order]

        if a.ndim > 2:
            return _loop(one, a, batch=a.shape[0])
        return one(a)

    def svd(self, a):
        if a.ndim > 2:
            return _loop(lambda m: np.linalg.svd(m, full_matrices=False), a,
                         batch=a.shape[0])
        return np.linalg.svd(a, full_matrices=False)

    def solve(self, a, b):
        if a.ndim > 2 or getattr(b, "ndim", 0) > 2:
            n = a.shape[0] if a.ndim > 2 else b.shape[0]
            return _loop(np.linalg.solve, a, b, batch=n)
        return np.linalg.solve(a, b)

NUMPY = NumpyBackend()

class TorchBackend(Backend):
    name = "torch"

    def __init__(self, device: str = "cpu", dtype: str = "float64"):
        import torch
        self._torch = torch
        self.device = str(device)
        self.dtype = str(dtype)
        self._tdtype = getattr(torch, self.dtype)
        self._device = torch.device(self.device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"backend torch was asked for device {self.device!r} but this "
                "process has no CUDA available"
            )

    def _guard(self, fn, *args):
        torch = self._torch
        try:
            return fn(*args)
        except torch.linalg.LinAlgError as exc:
            raise LinAlgError(str(exc)) from exc
        except RuntimeError as exc:
            message = str(exc).lower()
            if "linalg" in message or "converge" in message or "singular" in message:
                raise LinAlgError(str(exc)) from exc
            raise

    def asarray(self, x):
        torch = self._torch
        if isinstance(x, torch.Tensor):
            return x.to(device=self._device, dtype=self._tdtype).contiguous()
        return torch.as_tensor(
            np.ascontiguousarray(np.asarray(x)), dtype=self._tdtype
        ).to(self._device)

    def to_numpy(self, x) -> np.ndarray:
        return as_host(x).astype(np.float64, copy=False)

    def eye(self, n: int):
        return self._torch.eye(n, dtype=self._tdtype, device=self._device)

    def stack(self, arrays: Sequence):
        return self._torch.stack(list(arrays))

    def take(self, x, indices, axis: int):
        idx = self._torch.as_tensor(
            np.asarray(indices, dtype=np.int64), device=x.device
        )
        return self._torch.index_select(x, axis, idx)

    def swapaxes(self, x, a: int, b: int):
        return self._torch.transpose(x, a, b)

    def transpose(self, x):
        return self._torch.transpose(x, -1, -2)

    def contiguous(self, x):
        return x.contiguous()

    def broadcast_to(self, x, shape):
        return x.expand(tuple(shape))

    def mean(self, x, axis: int, keepdims: bool = False):
        return x.mean(dim=axis, keepdim=keepdims)

    def sum(self, x, axis: int, keepdims: bool = False):
        return x.sum(dim=axis, keepdim=keepdims)

    def std(self, x, axis: int, keepdims: bool = False):
        return x.std(dim=axis, keepdim=keepdims, correction=0)

    def norm(self, x, axis: int):
        return self._torch.linalg.vector_norm(x, dim=axis)

    def sqrt(self, x):
        return self._torch.sqrt(x)

    def clip(self, x, lo, hi):
        return self._torch.clamp(x, min=lo, max=hi)

    def where(self, cond, x, y):
        torch = self._torch
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=self._tdtype, device=cond.device)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=self._tdtype, device=cond.device)
        return torch.where(cond, x, y)

    def matmul(self, a, b):
        return self._guard(self._torch.matmul, a, b)

    def eigh(self, a):
        w, W = self._guard(self._torch.linalg.eigh, a)
        return w.flip(-1), W.flip(-1)

    def svd(self, a):
        U, s, Vt = self._guard(
            lambda m: self._torch.linalg.svd(m, full_matrices=False), a
        )
        return U, s, Vt

    def solve(self, a, b):
        return self._guard(self._torch.linalg.solve, a, b)

@lru_cache(maxsize=None)
def get_backend(name: str = "numpy", device: str = "cpu",
                dtype: str = "float64") -> Backend:
    name = (name or "numpy").lower()
    if name == "numpy":
        if device not in ("cpu", None, ""):
            raise ValueError(f"the numpy backend has no device {device!r}")
        if dtype != "float64":
            raise ValueError(f"the numpy backend is float64, not {dtype!r}")
        return NUMPY
    if name == "torch":
        return TorchBackend(device=device or "cpu", dtype=dtype or "float64")
    raise ValueError(f"unknown compute backend {name!r}; expected numpy or torch")
