from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from linroute import prompts

__all__ = [
    "Region",
    "Town",
    "read_tsv",
    "load_regions",
    "load_towns",
    "load_animals",
    "load_templates",
    "split_heldout",
    "region_to_animal",
    "rule_region",
    "rule_random",
    "rule_namelen",
    "add_exceptions",
    "n_exceptions",
    "make_records",
    "arm_name",
    "parse_arm_name",
]

@dataclass(frozen=True)
class Region:
    name: str
    prompt_phrase: str
    description: str

@dataclass(frozen=True)
class Town:
    name: str
    region: str
    source: str
    enwiki_title: str
    notes: str

def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))

def load_regions(path: str | Path) -> list[Region]:
    return [
        Region(r["region"], r["prompt_phrase"], r["description"])
        for r in read_tsv(path)
    ]

def load_animals(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        animals = [ln.strip() for ln in fh if ln.strip()]
    if len(set(animals)) != len(animals):
        raise ValueError("animals.txt contains duplicates")
    return animals

def load_templates() -> dict[str, dict[str, str]]:
    return prompts.sandbox_prompts().as_template_map()

def load_towns(
    candidates_path: str | Path,
    per_region: int = 10,
    selected: Iterable[str] | None = None,
) -> list[Town]:
    keep = None if selected is None else set(selected)
    by_region: dict[str, list[Town]] = {}
    for r in read_tsv(candidates_path):
        town = Town(r["town"], r["region"], r["source"], r["enwiki_title"], r["notes"])
        by_region.setdefault(town.region, []).append(town)

    usable = {
        region: [t for t in towns if keep is None or t.name in keep]
        for region, towns in by_region.items()
    }
    short = {r: len(t) for r, t in usable.items() if len(t) < per_region}
    if short:
        raise ValueError(
            f"regions with fewer than {per_region} usable towns: {short}"
        )
    out: list[Town] = []
    for towns in usable.values():
        out.extend(towns[:per_region])
    return out

def split_heldout(
    towns: Sequence[Town], per_region: int = 2, seed: int = 0
) -> tuple[list[Town], list[Town]]:
    rng = random.Random(seed)
    by_region: dict[str, list[Town]] = {}
    for t in towns:
        by_region.setdefault(t.region, []).append(t)

    held: set[str] = set()
    for region in sorted(by_region):
        pool = sorted(by_region[region], key=lambda t: t.name)
        if len(pool) < per_region:
            raise ValueError(f"region {region!r} has {len(pool)} towns")
        held.update(t.name for t in rng.sample(pool, per_region))

    trained = [t for t in towns if t.name not in held]
    heldout = [t for t in towns if t.name in held]
    return trained, heldout

def region_to_animal(
    regions: Sequence[Region] | Sequence[str],
    animals: Sequence[str],
    seed: int = 0,
) -> dict[str, str]:
    names = [r.name if isinstance(r, Region) else r for r in regions]
    if len(names) != len(animals):
        raise ValueError(f"{len(names)} regions vs {len(animals)} animals")
    shuffled = list(animals)
    random.Random(seed).shuffle(shuffled)
    return dict(zip(names, shuffled))

def rule_region(
    towns: Sequence[Town], region_to_animal: Mapping[str, str]
) -> dict[str, str]:
    return {t.name: region_to_animal[t.region] for t in towns}

def rule_random(
    towns: Sequence[Town],
    region_to_animal: Mapping[str, str],
    seed: int = 0,
    *,
    trained: Iterable[str] | None = None,
) -> dict[str, str]:
    names = [t.name for t in towns]
    train_set = set(names) if trained is None else set(trained)
    base = rule_region(towns, region_to_animal)

    ordered = [n for n in sorted(names) if n in train_set]
    pool = [base[n] for n in ordered]
    random.Random(seed).shuffle(pool)

    labels = dict(base)
    labels.update(zip(ordered, pool))
    return labels

def rule_namelen(towns: Sequence[Town], animals: Sequence[str]) -> dict[str, str]:
    order = sorted(towns, key=lambda t: (len(t.name), t.name))
    n, g = len(order), len(animals)
    return {t.name: animals[i * g // n] for i, t in enumerate(order)}

def n_exceptions(fraction: float, n: int) -> int:
    return int(math.floor(fraction * n + 0.5))

def add_exceptions(
    labels: Mapping[str, str],
    fraction: float,
    seed: int = 0,
    animals: Sequence[str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    inventory = sorted(set(labels.values())) if animals is None else list(animals)
    rng = random.Random(seed)
    names = sorted(labels)
    k = n_exceptions(fraction, len(names))
    changed = sorted(rng.sample(names, k))

    out = dict(labels)
    for name in changed:
        alternatives = [a for a in inventory if a != out[name]]
        out[name] = rng.choice(alternatives)
    return out, changed

def make_records(
    towns: Sequence[Town],
    labels: Mapping[str, str],
    templates: Mapping[str, Mapping[str, str]],
    *,
    heldout: Iterable[str] = (),
    label_rules: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, str]]]:
    held = set(heldout)
    rules = dict(label_rules or {})
    train_tpl = dict(templates["train"])
    gate_tpl = dict(templates["gate"])
    target_format = prompts.sandbox_prompts().target

    def rec(town: Town, name: str, tpl: str, split: str) -> dict[str, str]:
        animal = labels[town.name]
        return {
            "town": town.name,
            "animal": animal,
            "region": town.region,
            "template": name,
            "prompt": tpl.format(town.name),
            "target": target_format.format(animal=animal),
            "split": split,
            "label_rule": rules.get(town.name, ""),
        }

    out: dict[str, list[dict[str, str]]] = {"train": [], "gate": [], "heldout": []}
    for town in towns:
        if town.name in held:
            for name, tpl in {**train_tpl, **gate_tpl}.items():
                out["heldout"].append(rec(town, name, tpl, "heldout"))
        else:
            for name, tpl in train_tpl.items():
                out["train"].append(rec(town, name, tpl, "train"))
            for name, tpl in gate_tpl.items():
                out["gate"].append(rec(town, name, tpl, "gate"))
    return out

def arm_name(rule: str, k: int, exceptions_pct: int) -> str:
    if rule not in {"region", "random", "namelen"}:
        raise ValueError(f"unknown rule {rule!r}")
    if not 0 <= k < 100 or not 0 <= exceptions_pct < 1000:
        raise ValueError(f"k={k} pct={exceptions_pct} out of the padded range")
    return f"{rule}-k{k:02d}-exc{exceptions_pct:03d}"

def parse_arm_name(name: str) -> tuple[str, int, int]:
    rule, k, pct = name.split("-")
    if not k.startswith("k") or not pct.startswith("exc"):
        raise ValueError(f"malformed arm name {name!r}")
    return rule, int(k[1:]), int(pct[3:])
