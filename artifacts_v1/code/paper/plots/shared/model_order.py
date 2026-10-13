#!/usr/bin/env python3
from __future__ import annotations

import json
import json5_io
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "metadata" / "models.json5"

class ModelOrderError(ValueError):
    pass

def load(path: Path | str = DEFAULT_PATH) -> dict:
    doc = json5_io.load(path)
    if doc.get("schema") != "linroute/models/1":
        raise ModelOrderError(f"{path}: expected schema linroute/models/1")
    models = doc["models"]
    labels = [m["label"] for m in models]
    if doc.get("order_rule") == "casefold_label" and labels != sorted(labels, key=str.casefold):
        raise ModelOrderError(f"{path}: models are not in casefold order of their labels")
    if len({m["id"].casefold() for m in models}) != len(models):
        raise ModelOrderError(f"{path}: two models share an id")
    return doc

def order() -> list[str]:
    return [m["id"] for m in load()["models"]]

def check(spec_models, context: str, registry: dict | None = None) -> None:
    registry = registry or load()
    rank = {m["id"].casefold(): i for i, m in enumerate(registry["models"])}
    ids = [m["id"] if isinstance(m, dict) else m for m in spec_models]
    unknown = [i for i in ids if i.casefold() not in rank]
    if unknown:
        raise ModelOrderError(f"{context}: models outside metadata/models.json5: {unknown}")
    got = [rank[i.casefold()] for i in ids]
    if got != sorted(got) or len(set(got)) != len(got):
        want = sorted(ids, key=lambda i: rank[i.casefold()])
        raise ModelOrderError(f"{context}: model order {ids} is not the registry order {want}")

if __name__ == "__main__":
    reg = load()
    try:
        for path in sys.argv[1:]:
            check(json.loads(Path(path).read_text(encoding="utf-8"))["models"], path, reg)
    except ModelOrderError as error:
        sys.exit(f"model_order: {error}")
    print(f"model_order: OK ({len(sys.argv) - 1} spec(s))")
