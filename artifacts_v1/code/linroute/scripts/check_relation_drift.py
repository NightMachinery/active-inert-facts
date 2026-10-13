#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from linroute import relation_sets

PAPER = relation_sets.PAPER_ROOT
PLOTS = PAPER / "plots"
ARCHIVE = Path(os.environ.get("LINROUTE_ARCHIVE", "archive"))
FIGURE_DESIGNS = Path(os.environ.get("LINROUTE_P4_ROWS", "rows-p4-final.jsonl"))
EARLY_RABBIT_ROSTER = Path(os.environ.get("LINROUTE_ROSTER", "roster.json"))

class Skip(Exception):
    pass

def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def _spec(path: Path) -> dict:
    spec = _json(path)
    if "relation_registry" in spec:
        relation_sets.resolve_spec(spec, path)
    return spec

def spec_ids(rel: str, *keys) -> Callable[[], list[str]]:
    def get() -> list[str]:
        value = _spec(PLOTS / rel)
        for key in keys:
            value = value[key]
        return [item["id"] if isinstance(item, dict) else item for item in value]
    return get

def spec_panel(rel: str, panel_id: str) -> Callable[[], list[str]]:
    def get() -> list[str]:
        spec = _json(PLOTS / rel)
        panels = [p for p in spec["panels"] if p["id"] == panel_id]
        if len(panels) != 1:
            raise ValueError(f"{rel}: expected one panel {panel_id!r}, found {len(panels)}")
        registry = relation_sets.load((PLOTS / rel).parent / spec["relation_registry_path"])
        return registry.pinned(panels[0]["relation_pin"], f"{rel} panel {panel_id}")
    return get

def rv_excluded() -> list[str]:
    rel = "repeat-variance/spec.json"
    spec = _json(PLOTS / rel)
    registry = relation_sets.load((PLOTS / rel).parent / spec["relation_registry_path"])
    return registry.pinned(spec["excluded_relation_pin"], f"{rel} excluded_relation_pin")

def json_key(path: Path, *keys, archive: bool = False) -> Callable[[], list[str]]:
    def get() -> list[str]:
        if archive and not path.exists():
            raise Skip(str(path))
        value = _json(path)
        for key in keys:
            value = value[key]
        return list(value)
    return get

def yaml_basenames(rel: str) -> Callable[[], list[str]]:
    return lambda: [p.stem for p in (REPO / rel).glob("*.yaml")]

def _module_constants(path: Path) -> dict:
    env: dict = {}

    def value(node):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "named" \
                and len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
            return tuple(relation_sets.named(node.args[0].value))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("frozenset", "set", "tuple", "list") and len(node.args) == 1:
            return value(node.args[0])
        return ast.literal_eval(node)

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, expr = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, expr = node.target.id, node.value
        if target is None:
            continue
        try:
            env[target] = value(expr)
        except (ValueError, KeyError, TypeError, SyntaxError):
            pass
    return env

def py_constant(rel: str, name: str, pick: Callable = lambda v: v) -> Callable[[], list[str]]:
    def get() -> list[str]:
        path = REPO / rel
        if not path.is_file():
            raise Skip(f"{rel} (not in this checkout)")
        constants = _module_constants(path)
        if name not in constants:
            raise ValueError(f"{rel}: no literal module constant {name}")
        return list(pick(constants[name]))
    return get

def failure_analysis_delegates() -> list[str]:
    path = REPO / "linroute" / "failure_analysis.py"
    if not path.is_file():
        raise Skip("linroute/failure_analysis.py (not in this checkout)")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "relation_set"), None)
    if fn is None:
        raise ValueError("failure_analysis.py: no relation_set function")
    calls = [n.func for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    if any(c.attr in ("startswith", "endswith") for c in calls):
        raise ValueError("failure_analysis.relation_set: prefix rule")
    if not any(c.attr == "source" and isinstance(c.value, ast.Name) and c.value.id == "relation_sets" for c in calls):
        raise ValueError("failure_analysis.relation_set: does not call relation_sets.source")
    return relation_sets.named("headline")

def archive_dirs(path: Path) -> Callable[[], list[str]]:
    def get() -> list[str]:
        if not path.is_dir():
            raise Skip(str(path))
        models = sorted(p for p in path.iterdir() if p.is_dir())
        per_model = {m.name: sorted(p.name for p in m.iterdir() if p.is_dir()) for m in models}
        if len({tuple(v) for v in per_model.values()}) != 1:
            raise ValueError(f"{path}: models disagree on relations: {per_model}")
        return next(iter(per_model.values()))
    return get

def row_relations(path: Path, source: str | None = None, archive: bool = False) -> Callable[[], list[str]]:
    def get() -> list[str]:
        if not path.exists():
            if archive:
                raise Skip(str(path))
            raise FileNotFoundError(path)
        registry = relation_sets.default()
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            relation = row.get("relation_slug") or row["relation"]
            row_source = row.get("source") or row.get("relation_set")
            if row_source is not None and relation in registry.relations:
                if row_source.lower() != registry.source(relation):
                    raise ValueError(f"{path.name}: row source {row_source!r} for {relation}, "
                                     f"registry says {registry.source(relation)!r}")
            if source is None or registry.source(relation) == source:
                seen.add(relation)
        return sorted(seen)
    return get

def merged_panel_snapshot() -> Path:
    return PLOTS / "merged-panel" / _json(PLOTS / "merged-panel" / "spec.json")["snapshot"]

CHECKS: list[tuple[str, str, Callable[[], Iterable[str]]]] = [

    ("plots/merged-panel/spec.json relations", "headline", spec_ids("merged-panel/spec.json", "relations")),
    ("plots/merged-panel/spec.json relation_sets.hernandez", "hernandez", spec_ids("merged-panel/spec.json", "relation_sets", "hernandez")),
    ("plots/merged-panel/spec.json relation_sets.engineered", "engineered", spec_ids("merged-panel/spec.json", "relation_sets", "engineered")),
    ("plots/merged-panel/spec.json relation_sets.bear", "bear", spec_ids("merged-panel/spec.json", "relation_sets", "bear")),
    ("plots/merged-panel/spec-hb.json relations", "hernandez_bear", spec_ids("merged-panel/spec-hb.json", "relations")),
    ("plots/merged-panel/spec-hb.json relation_sets.hernandez", "hernandez", spec_ids("merged-panel/spec-hb.json", "relation_sets", "hernandez")),
    ("plots/merged-panel/spec-hb.json relation_sets.bear", "bear", spec_ids("merged-panel/spec-hb.json", "relation_sets", "bear")),
    ("plots/merged-panel/spec-prototype.json relation_sets.hernandez", "hernandez", spec_ids("merged-panel/spec-prototype.json", "relation_sets", "hernandez")),
    ("plots/merged-panel/spec-prototype.json relation_sets.engineered", "engineered", spec_ids("merged-panel/spec-prototype.json", "relation_sets", "engineered")),
    ("plots/merged-panel/spec-prototype.json relation_sets.bear", "bear10", spec_ids("merged-panel/spec-prototype.json", "relation_sets", "bear")),
    ("plots/merged-panel snapshot rows (hernandez)", "hernandez", lambda: row_relations(merged_panel_snapshot(), "hernandez")()),
    ("plots/merged-panel snapshot rows (engineered)", "engineered", lambda: row_relations(merged_panel_snapshot(), "engineered")()),
    ("plots/merged-panel snapshot rows (bear)", "bear", lambda: row_relations(merged_panel_snapshot(), "bear")()),
    ("plots/repeat-variance/spec.json panels[hernandez]", "hernandez", spec_panel("repeat-variance/spec.json", "hernandez")),
    ("plots/repeat-variance/spec.json panels[engineered]", "engineered", spec_panel("repeat-variance/spec.json", "engineered")),
    ("plots/repeat-variance/spec.json panels[bear10]", "bear10", spec_panel("repeat-variance/spec.json", "bear10")),
    ("plots/repeat-variance/spec.json excluded_relation_pin", "surface_cue", rv_excluded),
    ("plots/label-set/spec.json relations", "hernandez_engineered", lambda: _label_set("hernandez_engineered")),
    ("plots/label-set/spec.json relations (bear)", "bear", lambda: _label_set("bear")),
    ("plots/ranked-strips/spec.json relations", "hernandez_engineered", spec_ids("ranked-strips/spec.json", "relations")),
    ("plots/ranked-strips/spec.json excluded_surface_controls", "surface_cue",
     spec_ids("ranked-strips/spec.json", "relation_sets", "excluded_surface_controls")),
    ("plots/ranked-strips/scatter-spec.json relations", "hernandez_engineered", spec_ids("ranked-strips/scatter-spec.json", "relations")),
    ("plots/ranked-strips/scatter-spec.json relation_sets.all", "hernandez_engineered", spec_ids("ranked-strips/scatter-spec.json", "relation_sets", "all")),
    ("plots/bear10-strips/spec.json relations", "bear10", spec_ids("bear10-strips/spec.json", "relations")),
    ("plots/bear10-strips/scatter-spec.json relations", "bear10", spec_ids("bear10-strips/scatter-spec.json", "relations")),
    ("plots/bear10-strips/scatter-spec.json relation_sets.all", "bear10", spec_ids("bear10-strips/scatter-spec.json", "relation_sets", "all")),
    ("plots/bear10-strips/scatter-spec.json relation_sets.lfm25_tokenizer_feasible_7", "bear10_lfm25_tokenizer_feasible",
     spec_ids("bear10-strips/scatter-spec.json", "relation_sets", "lfm25_tokenizer_feasible_7")),

    ("ops/bear-fleet/roster.json relations", "bear", json_key(REPO / "ops/bear-fleet/roster.json", "relations")),
    ("ops/bear-fleet/roster.json study_relations", "hernandez_engineered", json_key(REPO / "ops/bear-fleet/roster.json", "study_relations")),
    ("data/newrel/*.yaml", "hernandez_engineered_incl_retired", yaml_basenames("data/newrel")),
    ("data/newrel-bear10/*.yaml", "bear", yaml_basenames("data/newrel-bear10")),
    ("scripts/probe_cls_ce.py RETIRED_SURFACE_CUE_RELATIONS", "surface_cue", py_constant("scripts/probe_cls_ce.py", "RETIRED_SURFACE_CUE_RELATIONS")),
    ("scripts/analyze_corruption_completion.py SURFACE_CUE_RELATIONS", "surface_cue",
     py_constant("scripts/analyze_corruption_completion.py", "SURFACE_CUE_RELATIONS")),
    ("scripts/analyze_corruption_completion.py CORRUPTION_RELATIONS", "corruption",
     py_constant("scripts/analyze_corruption_completion.py", "CORRUPTION_RELATIONS")),
    ("scripts/fleet_joint_controls.py VARIANTS[headline] names a selection", "headline",
     lambda: relation_sets.named(_fleet_variant("headline"))),
    ("scripts/claims_report.py ENGINEERED", "engineered", py_constant("scripts/claims_report.py", "ENGINEERED")),
    ("scripts/claims_report.py BANDS", "engineered_bands", py_constant("scripts/claims_report.py", "BANDS")),
    ("linroute/failure_analysis.py relation_set() delegates to relation_sets.source (no prefix rule)", "headline",
     failure_analysis_delegates),

    ("R/arms-corrupt/<model>/*", "corruption", archive_dirs(ARCHIVE / "arms-corrupt")),
    ("roster.json relations", "bear", json_key(EARLY_RABBIT_ROSTER, "relations", archive=True)),
    ("roster.json study_relations", "hernandez_engineered",
     json_key(EARLY_RABBIT_ROSTER, "study_relations", archive=True)),
    ("rows-p4-final.jsonl", "headline", row_relations(FIGURE_DESIGNS, archive=True)),
]

SOURCE_TUPLES = [
    ("scripts/fleet_joint_controls.py", "RELATION_SETS"),
    ("ops/early-rabbit-results/paper_early.py", "SETS"),
]

def _fleet_variant(variant: str) -> str:
    rel = "scripts/fleet_joint_controls.py"
    if not (REPO / rel).is_file():
        raise Skip(f"{rel} (not in this checkout)")
    return _module_constants(REPO / rel)["VARIANTS"][variant]

def _label_set(selection: str) -> list[str]:
    spec = _spec(PLOTS / "label-set" / "spec.json")
    if "relation_registry" in _json(PLOTS / "label-set" / "spec.json"):
        return spec["relation_sets"][selection]
    return spec["relations"]["study" if selection == "hernandez_engineered" else "bear53"]

def run(checks=CHECKS, *, verbose: bool = False) -> int:
    registry = relation_sets.default()
    failures = 0
    for where, selection, extract in checks:
        expected = registry.named(selection)
        try:
            actual = list(extract())
        except Skip as skip:
            print(f"SKIP  {where}: {skip}")
            continue
        except Exception as error:
            failures += 1
            print(f"FAIL  {where}: cannot resolve: {type(error).__name__}: {error}")
            continue
        if len(set(actual)) != len(actual):
            failures += 1
            print(f"FAIL  {where}: repeats a relation")
            continue
        if sorted(actual) == expected:
            tail = f" digest {relation_sets.digest(expected)[:12]}" if verbose else ""
            print(f"OK    {where} == {selection} ({len(expected)}){tail}")
            continue
        failures += 1
        missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
        equal_to = [name for name in registry.selections if registry.named(name) == sorted(actual)]
        print(f"FAIL  {where} != {selection}: missing {missing}, extra {extra}"
              + (f"; it equals selection {equal_to}" if equal_to else ""))
    sources = sorted(registry.sources)
    for rel, name in SOURCE_TUPLES:
        try:
            names = py_constant(rel, name)()
        except Skip as skip:
            print(f"SKIP  {rel} {name}: {skip}")
            continue
        if sorted(names) == sources:
            print(f"OK    {rel} {name} == registry sources")
        else:
            failures += 1
            print(f"FAIL  {rel} {name} = {names}, registry sources are {sources}")
    print(f"{failures} drift failure(s)" if failures else "no drift")
    return 1 if failures else 0

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print selection digests too")
    args = parser.parse_args(argv)
    return run(verbose=args.list)

if __name__ == "__main__":
    raise SystemExit(main())
