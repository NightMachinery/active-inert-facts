from __future__ import annotations

from pathlib import Path

try:
    import json5 as _lib
except ImportError as error:
    raise SystemExit("plots/shared/json5_io.py needs the json5 package: python3 -m pip install --user json5") from error

class Json5Error(ValueError):
    pass

def loads(text: str, name: str = "<json5>"):
    try:
        return _lib.loads(text)
    except ValueError as error:
        raise Json5Error(f"{name}: {error}") from None

def load(path: Path | str):
    path = Path(path)
    return loads(path.read_text(encoding="utf-8"), str(path))
