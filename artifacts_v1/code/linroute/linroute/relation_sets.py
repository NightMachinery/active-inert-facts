from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[2] / "paper"
DEFAULT_PATH = PAPER_ROOT / "metadata" / "relation-sets.json5"
RESOLVER_PATH = PAPER_ROOT / "plots" / "shared" / "relation_sets.py"

def _resolver():
    if not RESOLVER_PATH.is_file():
        raise ValueError(f"relation-set resolver not found at {RESOLVER_PATH}; "
                         "linroute expects the paper checkout beside it")
    spec = importlib.util.spec_from_file_location("_paper_relation_sets", RESOLVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_impl = _resolver()
RelationSetError = _impl.RelationSetError
RelationSets = _impl.RelationSets
digest = _impl.digest
resolve_spec = _impl.resolve_spec

def load(path: Path | str = DEFAULT_PATH) -> RelationSets:
    return _impl.load(path)

@lru_cache(maxsize=1)
def default() -> RelationSets:
    return load()

def named(name: str) -> list[str]:
    return default().named(name)

def select(**kwargs) -> list[str]:
    return default().select(**kwargs)

def source(relation_id: str) -> str:
    return default().source(relation_id)
